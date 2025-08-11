"""
Unit tests for Config class.

Tests configuration management, environment variable handling,
and validation for all QA Operator components.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from orchestrator.core.config import Config
from orchestrator.core.exceptions import ValidationError


class TestConfig:
    """Test cases for Config class."""

    def test_default_config_creation(self):
        """Test creating config with default values."""
        config = Config()

        assert config.ci_mode is False  # Default when CI env var not set
        assert config.headless_mode is None
        assert config.log_level == "INFO"
        assert config.log_format == "text"
        assert config.model_provider == "mixed"
        assert config.openai_api_key is None
        assert config.ollama_base_url == "http://localhost:11434"

    @patch.dict(os.environ, {"CI": "true"})
    def test_ci_mode_detection(self):
        """Test CI mode detection from environment variable."""
        config = Config()

        assert config.ci_mode is True
        assert config.log_format == "json"  # Should switch to JSON in CI

    @patch.dict(os.environ, {"CI": "false"})
    def test_ci_mode_false(self):
        """Test CI mode when explicitly set to false."""
        config = Config()

        assert config.ci_mode is False

    @patch.dict(os.environ, {"QA_OPERATOR_HEADLESS": "true"})
    def test_headless_mode_override(self):
        """Test headless mode override via environment variable."""
        config = Config()

        assert config.headless_mode is True

    @patch.dict(
        os.environ,
        {
            "QA_OPERATOR_LOG_LEVEL": "DEBUG",
            "QA_OPERATOR_MODEL_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
            "OLLAMA_BASE_URL": "http://custom:11434",
        },
    )
    def test_environment_variable_override(self):
        """Test all environment variable overrides."""
        config = Config()

        assert config.log_level == "DEBUG"
        assert config.model_provider == "openai"
        assert config.openai_api_key == "test-key"
        assert config.ollama_base_url == "http://custom:11434"

    def test_from_env_class_method(self):
        """Test creating config from environment using class method."""
        with patch.dict(os.environ, {"CI": "true", "QA_OPERATOR_LOG_LEVEL": "ERROR"}):
            config = Config.from_env()

            assert config.ci_mode is True
            assert config.log_level == "ERROR"

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = Config()
        config.openai_api_key = "test-key"

        # Should not raise any exceptions
        config.validate()

    def test_validate_missing_openai_key_mixed_provider(self):
        """Test validation fails when OpenAI key missing for mixed provider."""
        config = Config()
        config.model_provider = "mixed"
        config.openai_api_key = None

        with pytest.raises(ValidationError, match="OpenAI API key is required"):
            config.validate()

    def test_validate_missing_openai_key_openai_provider(self):
        """Test validation fails when OpenAI key missing for OpenAI provider."""
        config = Config()
        config.model_provider = "openai"
        config.openai_api_key = None

        with pytest.raises(ValidationError, match="OpenAI API key is required"):
            config.validate()

    def test_validate_ollama_only_provider(self):
        """Test validation passes for Ollama-only provider without OpenAI key."""
        config = Config()
        config.model_provider = "ollama"
        config.openai_api_key = None

        # Should not raise any exceptions
        config.validate()

    def test_validate_invalid_log_level(self):
        """Test validation fails for invalid log level."""
        config = Config()
        config.log_level = "INVALID"

        with pytest.raises(ValidationError, match="Invalid log level"):
            config.validate()

    def test_validate_invalid_model_provider(self):
        """Test validation fails for invalid model provider."""
        config = Config()
        config.model_provider = "invalid"

        with pytest.raises(ValidationError, match="Invalid model provider"):
            config.validate()

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = Config()
        config.openai_api_key = "test-key"

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["model_provider"] == "mixed"
        assert config_dict["log_level"] == "INFO"
        assert "openai_api_key" not in config_dict  # Should be excluded for security

    def test_artifact_retention_days_default(self):
        """Test default artifact retention days based on CI mode."""
        # Local development
        config = Config()
        assert config.artifact_retention_days == 7

        # CI environment
        with patch.dict(os.environ, {"CI": "true"}):
            config = Config()
            assert config.artifact_retention_days == 30

    @patch.dict(os.environ, {"QA_OPERATOR_ARTIFACT_RETENTION_DAYS": "14"})
    def test_artifact_retention_days_override(self):
        """Test artifact retention days override via environment variable."""
        config = Config()
        assert config.artifact_retention_days == 14

    def test_get_effective_headless_mode(self):
        """Test getting effective headless mode based on CI and override."""
        # Default behavior - follow CI mode
        config = Config()
        assert config.get_effective_headless_mode() is False

        # CI mode should default to headless
        with patch.dict(os.environ, {"CI": "true"}):
            config = Config()
            assert config.get_effective_headless_mode() is True

        # Explicit override should take precedence
        with patch.dict(os.environ, {"QA_OPERATOR_HEADLESS": "true"}):
            config = Config()
            assert config.get_effective_headless_mode() is True

    def test_get_log_file_path(self):
        """Test getting log file path based on configuration."""
        config = Config()

        log_path = config.get_log_file_path()
        assert isinstance(log_path, Path)
        assert log_path.name == "qa-operator.log"
        assert "logs" in str(log_path)

    def test_get_debug_log_dir(self):
        """Test getting debug log directory path."""
        config = Config()

        debug_dir = config.get_debug_log_dir()
        assert isinstance(debug_dir, Path)
        assert debug_dir.name == "debug"
        assert "logs" in str(debug_dir)
