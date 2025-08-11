"""
Unit tests for main CLI interface.

Tests command-line interface for manual execution, health checks,
and diagnostic utilities.
"""

import json
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from orchestrator.cli import (
    cmd_run, cmd_health, cmd_version, cmd_init,
    create_main_parser, main, _command_exists
)
from orchestrator.core.config import Config


class TestCLICommands:
    """Test cases for CLI commands."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create mock config
        self.mock_config = Config()
        self.mock_config.project_root = self.temp_dir
        self.mock_config.e2e_dir = self.temp_dir / "e2e"
        self.mock_config.artifacts_dir = self.temp_dir / "artifacts"
        self.mock_config.logs_dir = self.temp_dir / "logs"
        self.mock_config.policies_dir = self.temp_dir / "policies"
        self.mock_config.mcp_config_path = self.temp_dir / "mcp.config.json"
        self.mock_config.openai_api_key = "sk-test"
        
        # Create required directories
        for dir_path in [self.mock_config.e2e_dir, self.mock_config.policies_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create MCP config
        mcp_config = {
            "mcpServers": {
                "playwright": {"command": "npx", "args": ["@playwright/mcp@latest"]},
                "filesystem": {"command": "node", "args": ["./servers/fs-mcp.js"]}
            }
        }
        with open(self.mock_config.mcp_config_path, 'w') as f:
            json.dump(mcp_config, f)

    @patch('orchestrator.cli.get_config_manager')
    @patch('orchestrator.cli.validate_config')
    @patch('orchestrator.cli.QAOperatorAgent')
    def test_cmd_run_success(self, mock_agent_class, mock_validate, mock_get_manager, capsys):
        """Test run command with successful execution."""
        # Setup mocks
        mock_manager = MagicMock()
        mock_manager.get_config.return_value = self.mock_config
        mock_get_manager.return_value = mock_manager
        mock_validate.return_value = ([], False)  # No errors
        
        mock_agent = MagicMock()
        mock_agent.run_workflow.return_value = {"success": True, "tests_passed": 5}
        mock_agent_class.return_value = mock_agent
        
        # Create args
        args = MagicMock()
        args.spec_file = None
        args.verbose = False
        
        # Run command
        result = cmd_run(args)
        
        # Verify
        assert result == 0
        captured = capsys.readouterr()
        assert "Starting QA Operator workflow" in captured.out
        assert "completed successfully" in captured.out

    @patch('orchestrator.cli.get_config_manager')
    @patch('orchestrator.cli.validate_config')
    def test_cmd_run_validation_failure(self, mock_validate, mock_get_manager, capsys):
        """Test run command with validation failure."""
        # Setup mocks
        mock_manager = MagicMock()
        mock_manager.get_config.return_value = self.mock_config
        mock_get_manager.return_value = mock_manager
        mock_validate.return_value = (["Test error"], True)  # Has errors
        
        # Create args
        args = MagicMock()
        args.spec_file = None
        args.verbose = False
        
        # Run command
        result = cmd_run(args)
        
        # Verify
        assert result == 1
        captured = capsys.readouterr()
        assert "Configuration validation failed" in captured.out

    @patch('orchestrator.cli.get_config_manager')
    @patch('orchestrator.cli.validate_config')
    @patch('orchestrator.cli.QAOperatorAgent')
    def test_cmd_run_with_spec_file(self, mock_agent_class, mock_validate, mock_get_manager, capsys):
        """Test run command with specification file."""
        # Create spec file
        spec_file = self.temp_dir / "test_spec.json"
        spec_data = {"name": "test", "description": "Test spec"}
        with open(spec_file, 'w') as f:
            json.dump(spec_data, f)
        
        # Setup mocks
        mock_manager = MagicMock()
        mock_manager.get_config.return_value = self.mock_config
        mock_get_manager.return_value = mock_manager
        mock_validate.return_value = ([], False)
        
        mock_agent = MagicMock()
        mock_agent.run_workflow.return_value = {"success": True}
        mock_agent_class.return_value = mock_agent
        
        # Create args
        args = MagicMock()
        args.spec_file = str(spec_file)
        args.verbose = False
        
        # Run command
        result = cmd_run(args)
        
        # Verify
        assert result == 0
        mock_agent.run_workflow.assert_called_once_with(spec_data)

    @patch('orchestrator.cli.get_config_manager')
    @patch('orchestrator.cli.validate_config')
    @patch('orchestrator.cli.QAOperatorAgent')
    def test_cmd_run_spec_file_not_found(self, mock_agent_class, mock_validate, mock_get_manager, capsys):
        """Test run command with non-existent spec file."""
        # Setup mocks
        mock_manager = MagicMock()
        mock_manager.get_config.return_value = self.mock_config
        mock_get_manager.return_value = mock_manager
        mock_validate.return_value = ([], False)
        
        # Create args
        args = MagicMock()
        args.spec_file = "nonexistent.json"
        args.verbose = False
        
        # Run command
        result = cmd_run(args)
        
        # Verify
        assert result == 1
        captured = capsys.readouterr()
        assert "Specification file not found" in captured.out

    @patch('orchestrator.cli.get_config_manager')
    @patch('orchestrator.cli.validate_config')
    @patch('orchestrator.cli.QAOperatorAgent')
    def test_cmd_run_workflow_failure(self, mock_agent_class, mock_validate, mock_get_manager, capsys):
        """Test run command with workflow failure."""
        # Setup mocks
        mock_manager = MagicMock()
        mock_manager.get_config.return_value = self.mock_config
        mock_get_manager.return_value = mock_manager
        mock_validate.return_value = ([], False)
        
        mock_agent = MagicMock()
        mock_agent.run_workflow.return_value = {"success": False, "error": "Test error"}
        mock_agent_class.return_value = mock_agent
        
        # Create args
        args = MagicMock()
        args.spec_file = None
        args.verbose = False
        
        # Run command
        result = cmd_run(args)
        
        # Verify
        assert result == 1
        captured = capsys.readouterr()
        assert "workflow failed" in captured.out
        assert "Test error" in captured.out

    @patch('orchestrator.cli.get_config_manager')
    @patch('orchestrator.cli.validate_config')
    @patch('orchestrator.cli._command_exists')
    def test_cmd_health_success(self, mock_command_exists, mock_validate, mock_get_manager, capsys):
        """Test health command with successful check."""
        # Setup mocks
        mock_manager = MagicMock()
        mock_manager.get_config.return_value = self.mock_config
        mock_get_manager.return_value = mock_manager
        mock_validate.return_value = ([], False)  # No errors
        mock_command_exists.return_value = True
        
        # Create args
        args = MagicMock()
        
        # Run command
        result = cmd_health(args)
        
        # Verify
        assert result == 0
        captured = capsys.readouterr()
        assert "QA Operator Health Check" in captured.out
        assert "Health check passed" in captured.out

    @patch('orchestrator.cli.get_config_manager')
    @patch('orchestrator.cli.validate_config')
    def test_cmd_health_with_errors(self, mock_validate, mock_get_manager, capsys):
        """Test health command with validation errors."""
        # Setup mocks
        mock_manager = MagicMock()
        mock_manager.get_config.return_value = self.mock_config
        mock_get_manager.return_value = mock_manager
        mock_validate.return_value = (["Test error"], True)  # Has errors
        
        # Create args
        args = MagicMock()
        
        # Run command
        result = cmd_health(args)
        
        # Verify
        assert result == 1
        captured = capsys.readouterr()
        assert "Health check failed" in captured.out

    def test_cmd_version_basic(self, capsys):
        """Test version command basic output."""
        args = MagicMock()
        args.verbose = False
        
        result = cmd_version(args)
        
        assert result == 0
        captured = capsys.readouterr()
        assert "QA Operator" in captured.out

    def test_cmd_version_verbose(self, capsys):
        """Test version command verbose output."""
        args = MagicMock()
        args.verbose = True
        
        result = cmd_version(args)
        
        assert result == 0
        captured = capsys.readouterr()
        assert "QA Operator" in captured.out
        assert "System Information" in captured.out
        assert "Python:" in captured.out

    @patch('orchestrator.cli.get_config_manager')
    def test_cmd_init_success(self, mock_get_manager, capsys):
        """Test init command successful initialization."""
        # Setup mocks
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager
        
        # Create args
        args = MagicMock()
        
        # Change to temp directory
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(self.temp_dir)
            
            # Run command
            result = cmd_init(args)
            
            # Verify
            assert result == 0
            captured = capsys.readouterr()
            assert "Initializing QA Operator" in captured.out
            assert "initialization complete" in captured.out
            
            # Check directories were created
            assert (self.temp_dir / "e2e").exists()
            assert (self.temp_dir / "artifacts").exists()
            assert (self.temp_dir / "logs").exists()
            assert (self.temp_dir / "policies").exists()
            
            # Check files were created
            assert (self.temp_dir / ".env.example").exists()
            assert (self.temp_dir / "qa-operator.config.example.json").exists()
            
        finally:
            os.chdir(original_cwd)

    def test_cmd_init_existing_files(self, capsys):
        """Test init command with existing files."""
        # Create existing files
        (self.temp_dir / "e2e").mkdir(exist_ok=True)
        (self.temp_dir / ".env.example").touch()
        
        # Create args
        args = MagicMock()
        
        # Change to temp directory
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(self.temp_dir)
            
            # Run command
            result = cmd_init(args)
            
            # Verify
            assert result == 0
            captured = capsys.readouterr()
            assert "already exists" in captured.out
            
        finally:
            os.chdir(original_cwd)

    def test_command_exists_true(self):
        """Test _command_exists with existing command."""
        # Test with a command that should exist on most systems
        result = _command_exists("python")
        assert isinstance(result, bool)
        # Note: We can't guarantee python exists, so we just check the type

    def test_command_exists_false(self):
        """Test _command_exists with non-existent command."""
        result = _command_exists("nonexistent-command-12345")
        assert result is False


class TestCLIParser:
    """Test cases for CLI argument parser."""

    def test_create_main_parser(self):
        """Test creating main parser."""
        parser = create_main_parser()
        
        assert parser.prog == "qa-operator"
        assert "QA Operator" in parser.description

    def test_parser_run_command(self):
        """Test parser with run command."""
        parser = create_main_parser()
        
        args = parser.parse_args(["run", "--spec-file", "test.json", "--verbose"])
        
        assert args.command == "run"
        assert args.spec_file == "test.json"
        assert args.verbose is True

    def test_parser_health_command(self):
        """Test parser with health command."""
        parser = create_main_parser()
        
        args = parser.parse_args(["health"])
        
        assert args.command == "health"

    def test_parser_version_command(self):
        """Test parser with version command."""
        parser = create_main_parser()
        
        args = parser.parse_args(["version", "--verbose"])
        
        assert args.command == "version"
        assert args.verbose is True

    def test_parser_init_command(self):
        """Test parser with init command."""
        parser = create_main_parser()
        
        args = parser.parse_args(["init"])
        
        assert args.command == "init"

    def test_parser_config_command(self):
        """Test parser with config command."""
        parser = create_main_parser()
        
        args = parser.parse_args(["config"])
        
        assert args.command == "config"

    def test_parser_global_verbose(self):
        """Test parser with global verbose flag."""
        parser = create_main_parser()
        
        args = parser.parse_args(["--verbose", "version"])
        
        # Global verbose flag is set at top level
        assert hasattr(args, 'verbose')
        assert args.command == "version"


class TestMainFunction:
    """Test cases for main CLI function."""

    @patch('orchestrator.cli.cmd_run')
    def test_main_with_run_command(self, mock_cmd_run):
        """Test main function with run command."""
        mock_cmd_run.return_value = 0
        
        result = main(["run"])
        
        assert result == 0
        mock_cmd_run.assert_called_once()

    @patch('orchestrator.cli.config_main')
    def test_main_with_config_command(self, mock_config_main):
        """Test main function with config command."""
        mock_config_main.return_value = 0
        
        result = main(["config", "validate"])
        
        assert result == 0
        mock_config_main.assert_called_once_with(["validate"])

    def test_main_no_command(self, capsys):
        """Test main function with no command."""
        result = main([])
        
        assert result == 1
        captured = capsys.readouterr()
        assert "usage:" in captured.out

    @patch('orchestrator.cli.cmd_health')
    def test_main_with_exception(self, mock_cmd_health):
        """Test main function with command that raises exception."""
        mock_cmd_health.side_effect = Exception("Test error")
        
        with pytest.raises(Exception):
            main(["health"])

    def test_main_invalid_command(self, capsys):
        """Test main function with invalid command."""
        with pytest.raises(SystemExit):
            main(["invalid"])
        
        captured = capsys.readouterr()
        assert "invalid choice" in captured.err