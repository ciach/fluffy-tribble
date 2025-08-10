"""
Logging configuration for QA Operator.

Provides structured JSON logging with file rotation and different output formats
for development and CI environments.
"""

import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from .config import Config


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""

    def __init__(self, workflow_id: str):
        super().__init__()
        self.workflow_id = workflow_id

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "component": record.name,
            "workflow_id": self.workflow_id,
            "message": record.getMessage(),
        }

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        if hasattr(record, "metadata"):
            log_entry["metadata"] = record.metadata

        # Add context fields
        for attr in ["test_name", "model_name", "mcp_server", "duration", "status"]:
            if hasattr(record, attr):
                log_entry[attr] = getattr(record, attr)

        return json.dumps(log_entry, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-readable formatter for development."""

    def __init__(self, workflow_id: str):
        super().__init__()
        self.workflow_id = workflow_id

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as human-readable text."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        # Base format
        message = f"[{timestamp}] {record.levelname:8} {record.name:20} | {record.getMessage()}"

        # Add workflow ID
        message += f" (workflow: {self.workflow_id[:8]})"

        # Add metadata if present
        if hasattr(record, "metadata") and record.metadata:
            metadata_str = " | ".join(f"{k}={v}" for k, v in record.metadata.items())
            message += f" | {metadata_str}"

        # Add exception if present
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)

        return message


def setup_logging(config: Config, workflow_id: str) -> logging.Logger:
    """
    Set up logging configuration based on environment and config.

    Args:
        config: Configuration object with logging settings
        workflow_id: Unique workflow identifier for log correlation

    Returns:
        Configured root logger
    """
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set log level
    log_level = getattr(logging, config.log_level)
    root_logger.setLevel(log_level)

    # Choose formatter based on environment
    if config.log_format == "json":
        formatter = StructuredFormatter(workflow_id)
    else:
        formatter = TextFormatter(workflow_id)

    # Console handler (always present)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # File handler for non-CI environments
    if not config.is_ci_mode:
        # Main log file with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            config.get_log_file_path(),
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)

        # Debug file handler if debug is enabled
        if config.debug_enabled:
            debug_handler = logging.handlers.RotatingFileHandler(
                config.get_debug_log_dir() / f"debug-{workflow_id[:8]}.log",
                maxBytes=50 * 1024 * 1024,  # 50MB
                backupCount=3,
                encoding="utf-8",
            )
            debug_handler.setFormatter(formatter)
            debug_handler.setLevel(logging.DEBUG)
            root_logger.addHandler(debug_handler)

    # Log configuration startup
    logger = logging.getLogger("qa_operator.logging")
    logger.info(
        "Logging configured",
        extra={
            "metadata": {
                "workflow_id": workflow_id,
                "log_level": config.log_level,
                "log_format": config.log_format,
                "ci_mode": config.is_ci_mode,
                "debug_enabled": config.debug_enabled,
            }
        },
    )

    return root_logger


def get_logger(name: str, **context) -> logging.Logger:
    """
    Get a logger with optional context.

    Args:
        name: Logger name (typically module name)
        **context: Additional context to include in log records

    Returns:
        Configured logger with context
    """
    logger = logging.getLogger(name)

    # Add context as extra fields if provided
    if context:

        class ContextAdapter(logging.LoggerAdapter):
            def process(self, msg, kwargs):
                if "extra" not in kwargs:
                    kwargs["extra"] = {}
                kwargs["extra"].update(self.extra)
                return msg, kwargs

        return ContextAdapter(logger, context)

    return logger


def log_performance(
    logger: logging.Logger, operation: str, duration: float, **metadata
):
    """
    Log performance metrics for operations.

    Args:
        logger: Logger instance
        operation: Name of the operation
        duration: Duration in seconds
        **metadata: Additional metadata to include
    """
    logger.info(
        f"Performance: {operation} completed in {duration:.2f}s",
        extra={"metadata": {"operation": operation, "duration": duration, **metadata}},
    )


def log_mcp_call(
    logger: logging.Logger,
    server: str,
    method: str,
    duration: float,
    success: bool,
    **metadata,
):
    """
    Log MCP server calls for debugging and monitoring.

    Args:
        logger: Logger instance
        server: MCP server name
        method: Method called
        duration: Call duration in seconds
        success: Whether the call succeeded
        **metadata: Additional metadata
    """
    level = logging.DEBUG if success else logging.WARNING
    status = "success" if success else "failed"

    logger.log(
        level,
        f"MCP call: {server}.{method} {status} in {duration:.3f}s",
        extra={
            "metadata": {
                "mcp_server": server,
                "method": method,
                "duration": duration,
                "success": success,
                **metadata,
            }
        },
    )


def log_model_call(
    logger: logging.Logger,
    model: str,
    task_type: str,
    duration: float,
    success: bool,
    **metadata,
):
    """
    Log AI model calls for monitoring and debugging.

    Args:
        logger: Logger instance
        model: Model name
        task_type: Type of task (planning, generation, analysis, etc.)
        duration: Call duration in seconds
        success: Whether the call succeeded
        **metadata: Additional metadata
    """
    level = logging.DEBUG if success else logging.WARNING
    status = "success" if success else "failed"

    logger.log(
        level,
        f"Model call: {model} {task_type} {status} in {duration:.2f}s",
        extra={
            "metadata": {
                "model_name": model,
                "task_type": task_type,
                "duration": duration,
                "success": success,
                **metadata,
            }
        },
    )


def log_performance(operation_name: str):
    """
    Decorator to log performance metrics for functions.
    
    Args:
        operation_name: Name of the operation being measured
    
    Returns:
        Decorator function
    """
    import time
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(f"{func.__module__}.{func.__name__}")
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.info(
                    f"{operation_name} completed",
                    extra={
                        "metadata": {
                            "operation": operation_name,
                            "duration": duration,
                            "success": True
                        }
                    }
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                logger.error(
                    f"{operation_name} failed",
                    extra={
                        "metadata": {
                            "operation": operation_name,
                            "duration": duration,
                            "success": False,
                            "error": str(e)
                        }
                    }
                )
                
                raise
        
        return wrapper
    return decorator
