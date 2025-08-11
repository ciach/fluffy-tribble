"""
Unit tests for ConfigValidator class.

Tests configuration validation, system requirements checking,
and validation result reporting.
"""

import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from orchestrator.core.config_validator import (
    ConfigValidator, ValidationResult, ValidationLevel,
    validate_config, print_validation_results
)
from orchestrator.core.config import Config


class TestConfigValidator:
    """Test cases for ConfigValidator class."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = Config()
        self.config.project_root = self.temp_dir
        self.config.e2e_dir = self.temp_dir / "e2e"
        self.config.artifacts_dir = self.temp_dir / "artifacts"
        self.config.logs_dir = self.temp_dir / "logs"
        self.config.policies_dir = self.temp_dir / "policies"
        self.config.mcp_config_path = self.temp_dir / "mcp.config.json"
        
        # Create required directories
        self.config.e2e_dir.mkdir(exist_ok=True)
        self.config.policies_dir.mkdir(exist_ok=True)
        
        # Create valid MCP config
        mcp_config = {
            "mcpServers": {
                "playwright": {"command": "npx", "args": ["@playwright/mcp@latest"]},
                "filesystem": {"command": "node", "args": ["./servers/fs-mcp.js"]}
            }
        }
        with open(self.config.mcp_config_path, 'w') as f:
            json.dump(mcp_config, f)

    def test_validate_basic_config_valid(self):
        """Test validation of valid basic configuration."""
        self.config.openai_api_key = "sk-test"
        validator = ConfigValidator(self.config)
        
        validator._validate_basic_config()
        
        errors = validator.get_errors()
        assert len(errors) == 0

    def test_validate_basic_config_invalid_log_level(self):
        """Test validation with invalid log level."""
        self.config.log_level = "INVALID"
        validator = ConfigValidator(self.config)
        
        validator._validate_basic_config()
        
        errors = validator.get_errors()
        assert len(errors) > 0
        assert any("Invalid log level" in error.message for error in errors)

    def test_validate_basic_config_invalid_model_provider(self):
        """Test validation with invalid model provider."""
        self.config.model_provider = "invalid"
        validator = ConfigValidator(self.config)
        
        validator._validate_basic_config()
        
        errors = validator.get_errors()
        assert len(errors) > 0
        assert any("Invalid model provider" in error.message for error in errors)

    def test_validate_basic_config_invalid_retention_days(self):
        """Test validation with invalid artifact retention days."""
        self.config.artifact_retention_days = 0
        validator = ConfigValidator(self.config)
        
        validator._validate_basic_config()
        
        errors = validator.get_errors()
        assert len(errors) > 0
        assert any("Invalid artifact retention days" in error.message for error in errors)

    def test_validate_basic_config_long_retention_warning(self):
        """Test validation warning for very long retention."""
        self.config.artifact_retention_days = 400
        validator = ConfigValidator(self.config)
        
        validator._validate_basic_config()
        
        warnings = validator.get_warnings()
        assert len(warnings) > 0
        assert any("Very long artifact retention" in warning.message for warning in warnings)

    def test_validate_environment_variables_missing_openai_key(self):
        """Test validation with missing OpenAI API key."""
        self.config.model_provider = "openai"
        self.config.openai_api_key = None
        validator = ConfigValidator(self.config)
        
        validator._validate_environment_variables()
        
        errors = validator.get_errors()
        assert len(errors) > 0
        assert any("OpenAI API key required" in error.message for error in errors)

    def test_validate_environment_variables_invalid_openai_key_format(self):
        """Test validation with invalid OpenAI API key format."""
        self.config.model_provider = "openai"
        self.config.openai_api_key = "invalid-key"
        validator = ConfigValidator(self.config)
        
        validator._validate_environment_variables()
        
        warnings = validator.get_warnings()
        assert len(warnings) > 0
        assert any("API key format appears invalid" in warning.message for warning in warnings)

    def test_validate_environment_variables_invalid_ollama_url(self):
        """Test validation with invalid Ollama URL."""
        self.config.model_provider = "ollama"
        self.config.ollama_base_url = "invalid-url"
        validator = ConfigValidator(self.config)
        
        validator._validate_environment_variables()
        
        errors = validator.get_errors()
        assert len(errors) > 0
        assert any("Invalid Ollama URL format" in error.message for error in errors)

    @patch.dict(os.environ, {"CI": "true", "QA_OPERATOR_HEADLESS": "false"})
    def test_validate_environment_variables_conflicts(self):
        """Test validation with conflicting environment variables."""
        validator = ConfigValidator(self.config)
        
        validator._validate_environment_variables()
        
        warnings = validator.get_warnings()
        assert len(warnings) > 0
        assert any("Conflicting environment variables" in warning.message for warning in warnings)

    def test_validate_directories_missing_required(self):
        """Test validation with missing required directories."""
        # Remove required directory
        self.config.e2e_dir.rmdir()
        validator = ConfigValidator(self.config)
        
        validator._validate_directories()
        
        errors = validator.get_errors()
        assert len(errors) > 0
        assert any("does not exist" in error.message for error in errors)

    def test_validate_directories_not_directory(self):
        """Test validation when path exists but is not a directory."""
        # Create file instead of directory
        self.config.e2e_dir.rmdir()
        self.config.e2e_dir.touch()
        validator = ConfigValidator(self.config)
        
        validator._validate_directories()
        
        errors = validator.get_errors()
        assert len(errors) > 0
        assert any("is not a directory" in error.message for error in errors)

    def test_validate_file_permissions_not_writable(self):
        """Test validation with non-writable directories."""
        # Create directory but make it non-writable
        self.config.artifacts_dir.mkdir(exist_ok=True)
        
        with patch('os.access', return_value=False):
            validator = ConfigValidator(self.config)
            validator._validate_file_permissions()
        
        errors = validator.get_errors()
        assert len(errors) > 0
        assert any("is not writable" in error.message for error in errors)

    def test_validate_mcp_configuration_missing_file(self):
        """Test validation with missing MCP configuration file."""
        self.config.mcp_config_path.unlink()
        validator = ConfigValidator(self.config)
        
        validator._validate_mcp_configuration()
        
        errors = validator.get_errors()
        assert len(errors) > 0
        assert any("MCP configuration file not found" in error.message for error in errors)

    def test_validate_mcp_configuration_invalid_json(self):
        """Test validation with invalid MCP JSON."""
        with open(self.config.mcp_config_path, 'w') as f:
            f.write("invalid json")
        
        validator = ConfigValidator(self.config)
        validator._validate_mcp_configuration()
        
        errors = validator.get_errors()
        assert len(errors) > 0
        assert any("Invalid MCP configuration file" in error.message for error in errors)

    def test_validate_mcp_configuration_missing_servers_section(self):
        """Test validation with missing mcpServers section."""
        with open(self.config.mcp_config_path, 'w') as f:
            json.dump({}, f)
        
        validator = ConfigValidator(self.config)
        validator._validate_mcp_configuration()
        
        errors = validator.get_errors()
        assert len(errors) > 0
        assert any("missing 'mcpServers' section" in error.message for error in errors)

    def test_validate_mcp_configuration_missing_required_server(self):
        """Test validation with missing required MCP server."""
        mcp_config = {
            "mcpServers": {
                "playwright": {"command": "npx", "args": ["@playwright/mcp@latest"]}
                # Missing filesystem server
            }
        }
        with open(self.config.mcp_config_path, 'w') as f:
            json.dump(mcp_config, f)
        
        validator = ConfigValidator(self.config)
        validator._validate_mcp_configuration()
        
        errors = validator.get_errors()
        assert len(errors) > 0
        assert any("Required MCP server 'filesystem' not configured" in error.message for error in errors)

    def test_validate_mcp_server_config_missing_fields(self):
        """Test validation of MCP server config with missing fields."""
        mcp_config = {
            "mcpServers": {
                "playwright": {"command": "npx"},  # Missing args
                "filesystem": {"args": ["./servers/fs-mcp.js"]}  # Missing command
            }
        }
        with open(self.config.mcp_config_path, 'w') as f:
            json.dump(mcp_config, f)
        
        validator = ConfigValidator(self.config)
        validator._validate_mcp_configuration()
        
        errors = validator.get_errors()
        assert len(errors) >= 2
        assert any("missing 'args' field" in error.message for error in errors)
        assert any("missing 'command' field" in error.message for error in errors)

    @patch('subprocess.run')
    def test_validate_system_requirements_missing_command(self, mock_run):
        """Test validation with missing system commands."""
        mock_run.side_effect = FileNotFoundError()
        validator = ConfigValidator(self.config)
        
        validator._validate_system_requirements()
        
        errors = validator.get_errors()
        assert len(errors) > 0
        assert any("Required command not found" in error.message for error in errors)

    def test_validate_python_dependencies_missing_package(self):
        """Test validation with missing Python packages."""
        validator = ConfigValidator(self.config)
        
        with patch('builtins.__import__', side_effect=ImportError()):
            validator._validate_python_dependencies()
        
        errors = validator.get_errors()
        assert len(errors) > 0
        assert any("Required Python package not found" in error.message for error in errors)

    def test_validate_security_settings_debug_logging(self):
        """Test security validation with debug logging."""
        self.config.log_level = "DEBUG"
        validator = ConfigValidator(self.config)
        
        validator._validate_security_settings()
        
        warnings = validator.get_warnings()
        assert len(warnings) > 0
        assert any("Debug logging enabled" in warning.message for warning in warnings)

    def test_validate_all(self):
        """Test running all validation checks."""
        self.config.openai_api_key = "sk-test"
        validator = ConfigValidator(self.config)
        
        results = validator.validate_all()
        
        assert isinstance(results, list)
        assert all(isinstance(result, ValidationResult) for result in results)

    def test_validate_quick(self):
        """Test running quick validation checks."""
        validator = ConfigValidator(self.config)
        
        results = validator.validate_quick()
        
        assert isinstance(results, list)
        # Quick validation should have fewer results than full validation
        full_results = validator.validate_all()
        assert len(results) <= len(full_results)

    def test_has_errors(self):
        """Test checking if validation has errors."""
        self.config.log_level = "INVALID"
        validator = ConfigValidator(self.config)
        
        validator._validate_basic_config()
        
        assert validator.has_errors()

    def test_get_errors_and_warnings(self):
        """Test getting errors and warnings separately."""
        self.config.log_level = "INVALID"  # Error
        self.config.artifact_retention_days = 400  # Warning
        validator = ConfigValidator(self.config)
        
        validator._validate_basic_config()
        
        errors = validator.get_errors()
        warnings = validator.get_warnings()
        
        assert len(errors) > 0
        assert len(warnings) > 0
        assert all(result.level == ValidationLevel.ERROR for result in errors)
        assert all(result.level == ValidationLevel.WARNING for result in warnings)


class TestValidationResult:
    """Test cases for ValidationResult class."""

    def test_validation_result_creation(self):
        """Test creating validation result."""
        result = ValidationResult(
            level=ValidationLevel.ERROR,
            category="test",
            message="Test error",
            suggestion="Fix it",
            details={"key": "value"}
        )
        
        assert result.level == ValidationLevel.ERROR
        assert result.category == "test"
        assert result.message == "Test error"
        assert result.suggestion == "Fix it"
        assert result.details == {"key": "value"}

    def test_validation_result_optional_fields(self):
        """Test validation result with optional fields."""
        result = ValidationResult(
            level=ValidationLevel.INFO,
            category="test",
            message="Test info"
        )
        
        assert result.suggestion is None
        assert result.details is None


class TestValidationFunctions:
    """Test cases for validation utility functions."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = Config()
        self.config.project_root = self.temp_dir
        self.config.e2e_dir = self.temp_dir / "e2e"
        self.config.policies_dir = self.temp_dir / "policies"
        self.config.mcp_config_path = self.temp_dir / "mcp.config.json"
        
        # Create required directories and files
        self.config.e2e_dir.mkdir(exist_ok=True)
        self.config.policies_dir.mkdir(exist_ok=True)
        
        mcp_config = {
            "mcpServers": {
                "playwright": {"command": "npx", "args": ["@playwright/mcp@latest"]},
                "filesystem": {"command": "node", "args": ["./servers/fs-mcp.js"]}
            }
        }
        with open(self.config.mcp_config_path, 'w') as f:
            json.dump(mcp_config, f)

    def test_validate_config_function_quick(self):
        """Test validate_config function with quick mode."""
        self.config.openai_api_key = "sk-test"
        
        results, has_errors = validate_config(self.config, quick=True)
        
        assert isinstance(results, list)
        assert isinstance(has_errors, bool)

    def test_validate_config_function_full(self):
        """Test validate_config function with full validation."""
        self.config.openai_api_key = "sk-test"
        
        results, has_errors = validate_config(self.config, quick=False)
        
        assert isinstance(results, list)
        assert isinstance(has_errors, bool)

    def test_validate_config_function_with_errors(self):
        """Test validate_config function with configuration errors."""
        self.config.log_level = "INVALID"
        
        results, has_errors = validate_config(self.config, quick=True)
        
        assert has_errors
        assert len(results) > 0

    def test_print_validation_results_no_results(self, capsys):
        """Test printing validation results with no results."""
        print_validation_results([])
        
        captured = capsys.readouterr()
        assert "Configuration validation passed" in captured.out

    def test_print_validation_results_with_errors(self, capsys):
        """Test printing validation results with errors."""
        results = [
            ValidationResult(
                level=ValidationLevel.ERROR,
                category="test",
                message="Test error",
                suggestion="Fix it"
            )
        ]
        
        print_validation_results(results)
        
        captured = capsys.readouterr()
        assert "Error(s) found" in captured.out
        assert "Test error" in captured.out
        assert "Fix it" in captured.out

    def test_print_validation_results_with_warnings(self, capsys):
        """Test printing validation results with warnings."""
        results = [
            ValidationResult(
                level=ValidationLevel.WARNING,
                category="test",
                message="Test warning",
                suggestion="Consider fixing"
            )
        ]
        
        print_validation_results(results)
        
        captured = capsys.readouterr()
        assert "Warning(s) found" in captured.out
        assert "Test warning" in captured.out
        assert "Consider fixing" in captured.out

    def test_print_validation_results_hide_info(self, capsys):
        """Test printing validation results hiding info messages."""
        results = [
            ValidationResult(
                level=ValidationLevel.INFO,
                category="test",
                message="Test info"
            )
        ]
        
        print_validation_results(results, show_info=False)
        
        captured = capsys.readouterr()
        assert "Test info" not in captured.out