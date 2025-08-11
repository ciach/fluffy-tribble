"""
Unit tests for logging configuration.

Tests structured JSON logging, file rotation, and different output formats
for development and CI environments.
"""

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from orchestrator.core.logging_config import (
    StructuredFormatter,
    setup_logging,
    get_logger,
    log_performance,
)
from orchestrator.core.config import Config


class TestStructuredFormatter:
    """Test cases for StructuredFormatter."""

    def test_format_basic_log_record(self):
        """Test formatting basic log record."""
        formatter = StructuredFormatter("test-workflow-123")

        record = logging.LogRecord(
            name="test.component",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        log_data = json.loads(formatted)

        assert log_data["level"] == "INFO"
        assert log_data["component"] == "test.component"
        assert log_data["workflow_id"] == "test-workflow-123"
        assert log_data["message"] == "Test message"
        assert "timestamp" in log_data

    def test_format_with_metadata(self):
        """Test formatting log record with metadata."""
        formatter = StructuredFormatter("test-workflow-123")

        record = logging.LogRecord(
            name="test.component",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Error occurred",
            args=(),
            exc_info=None,
        )
        record.metadata = {"test_name": "user_login", "duration": 2.5}

        formatted = formatter.format(record)
        log_data = json.loads(formatted)

        assert log_data["metadata"]["test_name"] == "user_login"
        assert log_data["metadata"]["duration"] == 2.5

    def test_format_with_exception(self):
        """Test formatting log record with exception information."""
        formatter = StructuredFormatter("test-workflow-123")

        try:
            raise ValueError("Test exception")
        except ValueError:
            record = logging.LogRecord(
                name="test.component",
                level=logging.ERROR,
                pathname="",
                lineno=0,
                msg="Exception occurred",
                args=(),
                exc_info=True,
            )

        formatted = formatter.format(record)
        log_data = json.loads(formatted)

        assert "exception" in log_data
        assert "ValueError: Test exception" in log_data["exception"]

    def test_format_with_context_fields(self):
        """Test formatting log record with context fields."""
        formatter = StructuredFormatter("test-workflow-123")

        record = logging.LogRecord(
            name="test.component",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test execution",
            args=(),
            exc_info=None,
        )
        record.test_name = "user_registration"
        record.model_name = "gpt-4"
        record.mcp_server = "playwright"
        record.duration = 1.23
        record.status = "passed"

        formatted = formatter.format(record)
        log_data = json.loads(formatted)

        assert log_data["test_name"] == "user_registration"
        assert log_data["model_name"] == "gpt-4"
        assert log_data["mcp_server"] == "playwright"
        assert log_data["duration"] == 1.23
        assert log_data["status"] == "passed"


class TestLoggingSetup:
    """Test cases for logging setup functions."""

    def test_setup_logging_development_mode(self):
        """Test logging setup for development environment."""
        config = Config()
        config.log_format = "text"
        config.log_level = "DEBUG"

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"

            with patch(
                "orchestrator.core.logging_config.Config.get_log_file_path",
                return_value=log_file,
            ):
                setup_logging(config, "test-workflow-123")

                # Test that logger is configured
                logger = logging.getLogger("test")
                assert logger.level == logging.DEBUG

    def test_setup_logging_ci_mode(self):
        """Test logging setup for CI environment."""
        config = Config()
        config.log_format = "json"
        config.log_level = "INFO"
        config.ci_mode = True

        setup_logging(config, "test-workflow-123")

        # Test that logger is configured for CI
        logger = logging.getLogger("test")
        assert logger.level == logging.INFO

    def test_get_logger(self):
        """Test getting logger instance."""
        logger = get_logger("test.component")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.component"

    def test_log_performance_decorator(self):
        """Test performance logging decorator."""

        @log_performance("test_operation")
        def test_function(x, y):
            return x + y

        with patch("orchestrator.core.logging_config.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            result = test_function(2, 3)

            assert result == 5
            mock_logger.info.assert_called()

            # Check that performance info was logged
            call_args = mock_logger.info.call_args
            assert "test_operation completed" in call_args[0][0]
            assert "duration" in call_args[1]["extra"]["metadata"]

    def test_log_performance_with_exception(self):
        """Test performance logging decorator with exception."""

        @log_performance("test_operation")
        def failing_function():
            raise ValueError("Test error")

        with patch("orchestrator.core.logging_config.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            with pytest.raises(ValueError):
                failing_function()

            # Check that error was logged
            mock_logger.error.assert_called()
            call_args = mock_logger.error.call_args
            assert "test_operation failed" in call_args[0][0]

    def test_setup_file_rotation(self):
        """Test file rotation setup."""
        config = Config()
        config.log_format = "text"

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"

            with patch(
                "orchestrator.core.logging_config.Config.get_log_file_path",
                return_value=log_file,
            ):
                setup_logging(config, "test-workflow-123")

                # Test that log file is created
                logger = logging.getLogger("test")
                logger.info("Test message")

                # File should exist after logging
                assert log_file.exists()

    def test_debug_logging_setup(self):
        """Test debug logging directory setup."""
        config = Config()
        config.log_level = "DEBUG"

        with tempfile.TemporaryDirectory() as temp_dir:
            debug_dir = Path(temp_dir) / "debug"

            with patch(
                "orchestrator.core.logging_config.Config.get_debug_log_dir",
                return_value=debug_dir,
            ):
                setup_logging(config, "test-workflow-123")

                # Debug directory should be created
                assert debug_dir.exists()

    def test_workflow_id_correlation(self):
        """Test workflow ID correlation in logs."""
        config = Config()
        config.log_format = "json"

        setup_logging(config, "test-workflow-456")

        # Create a logger and test workflow ID is included
        logger = get_logger("test.component")

        with patch("sys.stdout") as mock_stdout:
            logger.info("Test message")

            # Should have written JSON with workflow_id
            written_data = mock_stdout.write.call_args[0][0]
            if written_data.strip():  # Skip empty writes
                log_data = json.loads(written_data.strip())
                assert log_data.get("workflow_id") == "test-workflow-456"
