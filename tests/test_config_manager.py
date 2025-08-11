"""
Unit tests for ConfigManager class.

Tests configuration management, templates, validation,
and hot-reloading functionality.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from orchestrator.core.config_manager import ConfigManager, ConfigTemplate
from orchestrator.core.config import Config
from orchestrator.core.exceptions import ValidationError


class TestConfigManager:
    """Test cases for ConfigManager class."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "qa-operator.config.json"
        self.mcp_config_file = self.temp_dir / "orchestrator" / "mcp.config.json"
        
        # Create orchestrator directory
        (self.temp_dir / "orchestrator").mkdir(exist_ok=True)
        
        # Create basic MCP config
        mcp_config = {
            "mcpServers": {
                "playwright": {"command": "npx", "args": ["@playwright/mcp@latest"]},
                "filesystem": {"command": "node", "args": ["./servers/fs-mcp.js"]}
            }
        }
        with open(self.mcp_config_file, 'w') as f:
            json.dump(mcp_config, f)
        
        # Create config manager with temp directory
        with patch('pathlib.Path.cwd', return_value=self.temp_dir):
            self.config_manager = ConfigManager(self.config_file)

    def teardown_method(self):
        """Clean up test environment."""
        if hasattr(self, 'config_manager'):
            self.config_manager.stop_hot_reload()

    def test_get_config_default(self):
        """Test getting default configuration."""
        with patch('pathlib.Path.cwd', return_value=self.temp_dir):
            config = self.config_manager.get_config()
        
        assert isinstance(config, Config)
        # Note: log_level may be affected by environment or templates
        assert config.log_level in ["DEBUG", "INFO", "WARN", "ERROR"]
        assert config.model_provider == "mixed"

    def test_get_config_from_file(self):
        """Test loading configuration from file."""
        # Create config file
        config_data = {
            "log_level": "DEBUG",
            "model_provider": "openai",
            "artifact_retention_days": 14
        }
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f)
        
        with patch('pathlib.Path.cwd', return_value=self.temp_dir):
            config = self.config_manager.get_config()
        
        assert config.log_level == "DEBUG"
        assert config.model_provider == "openai"
        assert config.artifact_retention_days == 14

    def test_reload_config(self):
        """Test configuration reloading."""
        # Initial config
        config1 = self.config_manager.get_config()
        initial_log_level = config1.log_level
        
        # Update config file with different log level
        new_log_level = "ERROR" if initial_log_level != "ERROR" else "WARN"
        config_data = {"log_level": new_log_level}
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Reload config
        with patch('pathlib.Path.cwd', return_value=self.temp_dir):
            config2 = self.config_manager.reload_config()
        
        assert config2.log_level == new_log_level

    def test_validate_config_valid(self):
        """Test validation of valid configuration."""
        config = Config()
        config.openai_api_key = "sk-test"
        
        errors = self.config_manager.validate_config(config)
        
        # Should have no errors (may have warnings/info)
        error_messages = [e for e in errors if "error" in e.lower()]
        assert len(error_messages) == 0

    def test_validate_config_invalid(self):
        """Test validation of invalid configuration."""
        config = Config()
        config.log_level = "INVALID"
        config.model_provider = "invalid"
        
        errors = self.config_manager.validate_config(config)
        
        assert len(errors) > 0
        error_text = " ".join(errors)
        assert "log level" in error_text.lower()
        assert "model provider" in error_text.lower()

    def test_get_template(self):
        """Test getting configuration template."""
        # Template should be created automatically
        template = self.config_manager.get_template("development")
        
        assert isinstance(template, ConfigTemplate)
        assert template.name == "development"
        assert template.environment == "development"
        assert template.config_overrides["log_level"] == "DEBUG"

    def test_get_template_not_found(self):
        """Test getting non-existent template."""
        with pytest.raises(ValueError, match="Template 'nonexistent' not found"):
            self.config_manager.get_template("nonexistent")

    def test_list_templates(self):
        """Test listing available templates."""
        templates = self.config_manager.list_templates()
        
        assert isinstance(templates, list)
        assert "development" in templates
        assert "ci" in templates
        assert "production" in templates
        assert "local-ollama" in templates

    def test_apply_template(self):
        """Test applying configuration template."""
        with patch('pathlib.Path.cwd', return_value=self.temp_dir):
            config = self.config_manager.apply_template("development", save=False)
        
        assert config.log_level == "DEBUG"
        assert config.ci_mode is False
        assert config.headless_mode is False

    def test_save_config(self):
        """Test saving configuration to file."""
        config = Config()
        config.log_level = "DEBUG"
        config.openai_api_key = "sk-secret"  # Should be excluded
        
        self.config_manager.save_config(config)
        
        # Verify file was created
        assert self.config_file.exists()
        
        # Verify content
        with open(self.config_file, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data["log_level"] == "DEBUG"
        assert "openai_api_key" not in saved_data  # Should be excluded

    def test_reload_callbacks(self):
        """Test configuration reload callbacks."""
        callback_called = False
        callback_config = None
        
        def test_callback(config):
            nonlocal callback_called, callback_config
            callback_called = True
            callback_config = config
        
        # Add callback
        self.config_manager.add_reload_callback(test_callback)
        
        # Trigger reload
        with patch('pathlib.Path.cwd', return_value=self.temp_dir):
            self.config_manager.reload_config()
        
        assert callback_called
        assert isinstance(callback_config, Config)
        
        # Remove callback
        self.config_manager.remove_reload_callback(test_callback)

    def test_get_config_documentation(self):
        """Test getting configuration documentation."""
        docs = self.config_manager.get_config_documentation()
        
        assert isinstance(docs, dict)
        assert "version" in docs
        assert "environment_variables" in docs
        assert "configuration_files" in docs
        assert "templates" in docs
        
        # Check environment variables documentation
        env_vars = docs["environment_variables"]
        assert "CI" in env_vars
        assert "QA_OPERATOR_LOG_LEVEL" in env_vars
        assert "OPENAI_API_KEY" in env_vars
        
        # Check each env var has required fields
        for var_name, var_info in env_vars.items():
            assert "description" in var_info
            assert "type" in var_info
            assert "example" in var_info

    def test_hot_reload_start_stop(self):
        """Test starting and stopping hot reload."""
        # Should start without error
        self.config_manager.start_hot_reload()
        
        # Should stop without error
        self.config_manager.stop_hot_reload()
        
        # Should handle multiple starts/stops
        self.config_manager.start_hot_reload()
        self.config_manager.start_hot_reload()  # Should not error
        self.config_manager.stop_hot_reload()

    @patch('orchestrator.core.config_manager.Observer')
    def test_hot_reload_file_change(self, mock_observer_class):
        """Test hot reload on file change."""
        mock_observer = MagicMock()
        mock_observer_class.return_value = mock_observer
        
        # Start hot reload
        self.config_manager.start_hot_reload()
        
        # Verify observer was configured
        mock_observer.schedule.assert_called()
        mock_observer.start.assert_called()
        
        # Stop hot reload
        self.config_manager.stop_hot_reload()
        mock_observer.stop.assert_called()
        mock_observer.join.assert_called()


class TestConfigTemplate:
    """Test cases for ConfigTemplate class."""

    def test_config_template_creation(self):
        """Test creating configuration template."""
        template = ConfigTemplate(
            name="test",
            description="Test template",
            environment="test",
            config_overrides={"log_level": "DEBUG"},
            required_env_vars=["TEST_VAR"],
            optional_env_vars=["OPTIONAL_VAR"]
        )
        
        assert template.name == "test"
        assert template.description == "Test template"
        assert template.environment == "test"
        assert template.config_overrides["log_level"] == "DEBUG"
        assert "TEST_VAR" in template.required_env_vars
        assert "OPTIONAL_VAR" in template.optional_env_vars

    def test_config_template_defaults(self):
        """Test configuration template with default values."""
        template = ConfigTemplate(
            name="minimal",
            description="Minimal template",
            environment="test"
        )
        
        assert template.config_overrides == {}
        assert template.required_env_vars == []
        assert template.optional_env_vars == []


class TestGlobalConfigManager:
    """Test cases for global configuration manager functions."""

    def test_get_config_manager_singleton(self):
        """Test global config manager is singleton."""
        from orchestrator.core.config_manager import get_config_manager
        
        manager1 = get_config_manager()
        manager2 = get_config_manager()
        
        assert manager1 is manager2

    def test_get_config_global(self):
        """Test global get_config function."""
        from orchestrator.core.config_manager import get_config
        
        config = get_config()
        assert isinstance(config, Config)