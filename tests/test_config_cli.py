"""
Unit tests for configuration CLI utilities.

Tests command-line interface for configuration management,
validation, and health checks.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from orchestrator.core.config_cli import (
    cmd_validate, cmd_show, cmd_templates, cmd_health, cmd_docs,
    create_config_parser, main
)
from orchestrator.core.config import Config
from orchestrator.core.config_validator import ValidationResult, ValidationLevel


class TestConfigCLI:
    """Test cases for configuration CLI commands."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create mock config
        self.mock_config = Config()
        self.mock_config.project_root = self.temp_dir
        self.mock_config.openai_api_key = "sk-test"

    @patch('orchestrator.core.config_cli.get_config_manager')
    @patch('orchestrator.core.config_cli.validate_config')
    def test_cmd_validate_success(self, mock_validate, mock_get_manager, capsys):
        """Test validate command with successful validation."""
        # Setup mocks
        mock_manager = MagicMock()
        mock_manager.get_config.return_value = self.mock_config
        mock_get_manager.return_value = mock_manager
        mock_validate.return_value = ([], False)  # No errors
        
        # Create args
        args = MagicMock()
        args.quick = False
        args.quiet = False
        
        # Run command
        result = cmd_validate(args)
        
        # Verify
        assert result == 0
        captured = capsys.readouterr()
        assert "Validating QA Operator configuration" in captured.out

    @patch('orchestrator.core.config_cli.get_config_manager')
    @patch('orchestrator.core.config_cli.validate_config')
    def test_cmd_validate_with_errors(self, mock_validate, mock_get_manager, capsys):
        """Test validate command with validation errors."""
        # Setup mocks
        mock_manager = MagicMock()
        mock_manager.get_config.return_value = self.mock_config
        mock_get_manager.return_value = mock_manager
        
        error_result = ValidationResult(
            level=ValidationLevel.ERROR,
            category="test",
            message="Test error"
        )
        mock_validate.return_value = ([error_result], True)  # Has errors
        
        # Create args
        args = MagicMock()
        args.quick = False
        args.quiet = False
        
        # Run command
        result = cmd_validate(args)
        
        # Verify
        assert result == 1

    @patch('orchestrator.core.config_cli.get_config_manager')
    def test_cmd_validate_exception(self, mock_get_manager, capsys):
        """Test validate command with exception."""
        # Setup mock to raise exception
        mock_get_manager.side_effect = Exception("Test error")
        
        # Create args
        args = MagicMock()
        args.quick = False
        args.quiet = False
        
        # Run command
        result = cmd_validate(args)
        
        # Verify
        assert result == 1
        captured = capsys.readouterr()
        assert "Configuration validation failed" in captured.out

    @patch('orchestrator.core.config_cli.get_config_manager')
    def test_cmd_show_text_format(self, mock_get_manager, capsys):
        """Test show command with text format."""
        # Setup mocks
        mock_manager = MagicMock()
        mock_manager.get_config.return_value = self.mock_config
        mock_get_manager.return_value = mock_manager
        
        # Create args
        args = MagicMock()
        args.format = "text"
        
        # Run command
        result = cmd_show(args)
        
        # Verify
        assert result == 0
        captured = capsys.readouterr()
        assert "Current QA Operator Configuration" in captured.out
        assert "Basic Settings" in captured.out
        assert "AI Model Settings" in captured.out

    @patch('orchestrator.core.config_cli.get_config_manager')
    def test_cmd_show_json_format(self, mock_get_manager, capsys):
        """Test show command with JSON format."""
        # Setup mocks
        mock_manager = MagicMock()
        mock_manager.get_config.return_value = self.mock_config
        mock_get_manager.return_value = mock_manager
        
        # Create args
        args = MagicMock()
        args.format = "json"
        
        # Run command
        result = cmd_show(args)
        
        # Verify
        assert result == 0
        captured = capsys.readouterr()
        
        # Should be valid JSON
        try:
            json.loads(captured.out)
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")

    @patch('orchestrator.core.config_cli.get_config_manager')
    def test_cmd_templates_list(self, mock_get_manager, capsys):
        """Test templates command with list action."""
        # Setup mocks
        mock_manager = MagicMock()
        mock_manager.list_templates.return_value = ["development", "ci"]
        
        mock_template = MagicMock()
        mock_template.description = "Test template"
        mock_template.environment = "test"
        mock_template.required_env_vars = ["TEST_VAR"]
        mock_manager.get_template.return_value = mock_template
        
        mock_get_manager.return_value = mock_manager
        
        # Create args
        args = MagicMock()
        args.action = "list"
        
        # Run command
        result = cmd_templates(args)
        
        # Verify
        assert result == 0
        captured = capsys.readouterr()
        assert "Available Configuration Templates" in captured.out
        assert "development" in captured.out
        assert "ci" in captured.out

    @patch('orchestrator.core.config_cli.get_config_manager')
    def test_cmd_templates_list_empty(self, mock_get_manager, capsys):
        """Test templates command with no templates."""
        # Setup mocks
        mock_manager = MagicMock()
        mock_manager.list_templates.return_value = []
        mock_get_manager.return_value = mock_manager
        
        # Create args
        args = MagicMock()
        args.action = "list"
        
        # Run command
        result = cmd_templates(args)
        
        # Verify
        assert result == 0
        captured = capsys.readouterr()
        assert "No configuration templates found" in captured.out

    @patch('orchestrator.core.config_cli.get_config_manager')
    @patch('orchestrator.core.config_cli.validate_config')
    def test_cmd_templates_apply(self, mock_validate, mock_get_manager, capsys):
        """Test templates command with apply action."""
        # Setup mocks
        mock_manager = MagicMock()
        mock_manager.apply_template.return_value = self.mock_config
        mock_get_manager.return_value = mock_manager
        mock_validate.return_value = ([], False)  # No errors
        
        # Create args
        args = MagicMock()
        args.action = "apply"
        args.template = "development"
        args.dry_run = False
        
        # Run command
        result = cmd_templates(args)
        
        # Verify
        assert result == 0
        captured = capsys.readouterr()
        assert "Applying configuration template" in captured.out
        assert "applied successfully" in captured.out

    @patch('orchestrator.core.config_cli.get_config_manager')
    def test_cmd_templates_apply_dry_run(self, mock_get_manager, capsys):
        """Test templates command with dry run."""
        # Setup mocks
        mock_manager = MagicMock()
        mock_manager.apply_template.return_value = self.mock_config
        mock_get_manager.return_value = mock_manager
        
        # Create args
        args = MagicMock()
        args.action = "apply"
        args.template = "development"
        args.dry_run = True
        
        # Run command
        result = cmd_templates(args)
        
        # Verify
        assert result == 0
        captured = capsys.readouterr()
        assert "Dry run" in captured.out

    @patch('orchestrator.core.config_cli.get_config_manager')
    def test_cmd_templates_apply_no_template(self, mock_get_manager, capsys):
        """Test templates command apply without template name."""
        # Create args
        args = MagicMock()
        args.action = "apply"
        args.template = None
        
        # Run command
        result = cmd_templates(args)
        
        # Verify
        assert result == 1
        captured = capsys.readouterr()
        assert "Template name required" in captured.out

    @patch('orchestrator.core.config_cli.get_config_manager')
    def test_cmd_templates_apply_invalid_template(self, mock_get_manager, capsys):
        """Test templates command with invalid template."""
        # Setup mocks
        mock_manager = MagicMock()
        mock_manager.apply_template.side_effect = ValueError("Template not found")
        mock_get_manager.return_value = mock_manager
        
        # Create args
        args = MagicMock()
        args.action = "apply"
        args.template = "nonexistent"
        args.dry_run = False
        
        # Run command
        result = cmd_templates(args)
        
        # Verify
        assert result == 1
        captured = capsys.readouterr()
        assert "Template error" in captured.out

    @patch('orchestrator.core.config_cli.get_config_manager')
    @patch('orchestrator.core.config_cli.validate_config')
    def test_cmd_health_success(self, mock_validate, mock_get_manager, capsys):
        """Test health command with successful check."""
        # Setup mocks
        mock_manager = MagicMock()
        mock_manager.get_config.return_value = self.mock_config
        mock_get_manager.return_value = mock_manager
        mock_validate.return_value = ([], False)  # No errors
        
        # Create required directories
        self.mock_config.e2e_dir.mkdir(exist_ok=True)
        self.mock_config.policies_dir.mkdir(exist_ok=True)
        
        # Create MCP config
        mcp_config = {
            "mcpServers": {
                "playwright": {"command": "npx", "args": ["@playwright/mcp@latest"]},
                "filesystem": {"command": "node", "args": ["./servers/fs-mcp.js"]}
            }
        }
        with open(self.mock_config.mcp_config_path, 'w') as f:
            json.dump(mcp_config, f)
        
        # Create args
        args = MagicMock()
        
        # Run command
        result = cmd_health(args)
        
        # Verify
        assert result == 0
        captured = capsys.readouterr()
        assert "QA Operator Health Check" in captured.out
        assert "Health check passed" in captured.out

    @patch('orchestrator.core.config_cli.get_config_manager')
    @patch('orchestrator.core.config_cli.validate_config')
    def test_cmd_health_with_errors(self, mock_validate, mock_get_manager, capsys):
        """Test health command with validation errors."""
        # Setup mocks
        mock_manager = MagicMock()
        mock_manager.get_config.return_value = self.mock_config
        mock_get_manager.return_value = mock_manager
        
        error_result = ValidationResult(
            level=ValidationLevel.ERROR,
            category="test",
            message="Test error"
        )
        mock_validate.return_value = ([error_result], True)  # Has errors
        
        # Create args
        args = MagicMock()
        
        # Run command
        result = cmd_health(args)
        
        # Verify
        assert result == 1
        captured = capsys.readouterr()
        assert "Health check failed" in captured.out

    @patch('orchestrator.core.config_cli.get_config_manager')
    def test_cmd_docs_text_format(self, mock_get_manager, capsys):
        """Test docs command with text format."""
        # Setup mocks
        mock_manager = MagicMock()
        mock_docs = {
            "version": "1.0.0",
            "environment_variables": {
                "CI": {
                    "description": "CI mode",
                    "type": "boolean",
                    "default": "false",
                    "example": "true"
                }
            },
            "configuration_files": {
                "qa-operator.config.json": {
                    "description": "Main config",
                    "location": "/path/to/config",
                    "format": "JSON",
                    "hot_reload": True
                }
            },
            "templates": {
                "development": {
                    "description": "Dev template",
                    "environment": "development"
                }
            }
        }
        mock_manager.get_config_documentation.return_value = mock_docs
        mock_get_manager.return_value = mock_manager
        
        # Create args
        args = MagicMock()
        args.format = "text"
        
        # Run command
        result = cmd_docs(args)
        
        # Verify
        assert result == 0
        captured = capsys.readouterr()
        assert "Configuration Documentation" in captured.out
        assert "Environment Variables" in captured.out
        assert "Configuration Files" in captured.out

    @patch('orchestrator.core.config_cli.get_config_manager')
    def test_cmd_docs_json_format(self, mock_get_manager, capsys):
        """Test docs command with JSON format."""
        # Setup mocks
        mock_manager = MagicMock()
        mock_docs = {"version": "1.0.0"}
        mock_manager.get_config_documentation.return_value = mock_docs
        mock_get_manager.return_value = mock_manager
        
        # Create args
        args = MagicMock()
        args.format = "json"
        
        # Run command
        result = cmd_docs(args)
        
        # Verify
        assert result == 0
        captured = capsys.readouterr()
        
        # Should be valid JSON
        try:
            parsed = json.loads(captured.out)
            assert parsed["version"] == "1.0.0"
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")


class TestConfigParser:
    """Test cases for configuration argument parser."""

    def test_create_config_parser(self):
        """Test creating configuration parser."""
        parser = create_config_parser()
        
        assert parser.prog == "qa-operator config"
        assert "Configuration commands" in parser.description

    def test_parser_validate_command(self):
        """Test parser with validate command."""
        parser = create_config_parser()
        
        args = parser.parse_args(["validate", "--quick", "--quiet"])
        
        assert args.command == "validate"
        assert args.quick is True
        assert args.quiet is True

    def test_parser_show_command(self):
        """Test parser with show command."""
        parser = create_config_parser()
        
        args = parser.parse_args(["show", "--format", "json"])
        
        assert args.command == "show"
        assert args.format == "json"

    def test_parser_templates_command(self):
        """Test parser with templates command."""
        parser = create_config_parser()
        
        args = parser.parse_args(["templates", "apply", "--template", "development", "--dry-run"])
        
        assert args.command == "templates"
        assert args.action == "apply"
        assert args.template == "development"
        assert args.dry_run is True

    def test_parser_health_command(self):
        """Test parser with health command."""
        parser = create_config_parser()
        
        args = parser.parse_args(["health"])
        
        assert args.command == "health"

    def test_parser_docs_command(self):
        """Test parser with docs command."""
        parser = create_config_parser()
        
        args = parser.parse_args(["docs", "--format", "json"])
        
        assert args.command == "docs"
        assert args.format == "json"


class TestMainFunction:
    """Test cases for main CLI function."""

    @patch('orchestrator.core.config_cli.cmd_validate')
    def test_main_with_validate_command(self, mock_cmd_validate):
        """Test main function with validate command."""
        mock_cmd_validate.return_value = 0
        
        result = main(["validate"])
        
        assert result == 0
        mock_cmd_validate.assert_called_once()

    def test_main_no_command(self, capsys):
        """Test main function with no command."""
        result = main([])
        
        assert result == 1
        captured = capsys.readouterr()
        assert "usage:" in captured.out

    @patch('orchestrator.core.config_cli.cmd_show')
    def test_main_with_exception(self, mock_cmd_show):
        """Test main function with command that raises exception."""
        mock_cmd_show.side_effect = Exception("Test error")
        
        with pytest.raises(Exception):
            main(["show"])

    def test_main_invalid_command(self, capsys):
        """Test main function with invalid command."""
        with pytest.raises(SystemExit):
            main(["invalid"])
        
        captured = capsys.readouterr()
        assert "invalid choice" in captured.err