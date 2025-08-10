"""
Configuration management for QA Operator.

Handles environment variables, defaults, and configuration validation
for all QA Operator components.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Configuration class for QA Operator with environment variable support."""

    # Environment detection
    ci_mode: bool = field(default_factory=lambda: os.getenv("CI", "").lower() == "true")

    # Execution settings
    headless_mode: Optional[bool] = field(
        default_factory=lambda: (
            True
            if os.getenv("QA_OPERATOR_HEADLESS", "").lower() == "true"
            else None  # Will be determined by CI mode if not explicitly set
        )
    )

    # Logging configuration
    log_level: str = field(
        default_factory=lambda: os.getenv("QA_OPERATOR_LOG_LEVEL", "INFO")
    )
    log_format: str = field(
        default_factory=lambda: (
            "json" if os.getenv("CI", "").lower() == "true" else "text"
        )
    )

    # Model configuration
    model_provider: str = field(
        default_factory=lambda: os.getenv("QA_OPERATOR_MODEL_PROVIDER", "mixed")
    )
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )

    # Artifact management
    artifact_retention_days: int = field(
        default_factory=lambda: int(
            os.getenv(
                "QA_OPERATOR_ARTIFACT_RETENTION_DAYS",
                "30" if os.getenv("CI", "").lower() == "true" else "7",
            )
        )
    )

    # Directory paths
    project_root: Path = field(default_factory=lambda: Path.cwd())
    e2e_dir: Path = field(default_factory=lambda: Path.cwd() / "e2e")
    artifacts_dir: Path = field(default_factory=lambda: Path.cwd() / "artifacts")
    logs_dir: Path = field(default_factory=lambda: Path.cwd() / "logs")
    policies_dir: Path = field(default_factory=lambda: Path.cwd() / "policies")

    # MCP configuration
    mcp_config_path: Path = field(
        default_factory=lambda: Path.cwd() / "orchestrator" / "mcp.config.json"
    )

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Determine headless mode if not explicitly set
        if self.headless_mode is None:
            self.headless_mode = self.ci_mode

        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARN", "ERROR"]
        if self.log_level.upper() not in valid_log_levels:
            self.log_level = "INFO"
        else:
            self.log_level = self.log_level.upper()

        # Validate model provider
        valid_providers = ["openai", "ollama", "mixed"]
        if self.model_provider not in valid_providers:
            self.model_provider = "mixed"

        # Ensure directories exist
        self.artifacts_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.e2e_dir.mkdir(exist_ok=True)

    @property
    def is_headless(self) -> bool:
        """Get effective headless mode setting."""
        return self.headless_mode

    @property
    def is_ci_mode(self) -> bool:
        """Check if running in CI environment."""
        return self.ci_mode

    @property
    def debug_enabled(self) -> bool:
        """Check if debug logging is enabled."""
        return self.log_level == "DEBUG"

    def get_log_file_path(self) -> Path:
        """Get the main log file path."""
        return self.logs_dir / "qa-operator.log"

    def get_debug_log_dir(self) -> Path:
        """Get the debug log directory path."""
        debug_dir = self.logs_dir / "debug"
        debug_dir.mkdir(exist_ok=True)
        return debug_dir

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for logging."""
        return {
            "ci_mode": self.ci_mode,
            "headless_mode": self.headless_mode,
            "log_level": self.log_level,
            "log_format": self.log_format,
            "model_provider": self.model_provider,
            "artifact_retention_days": self.artifact_retention_days,
            "project_root": str(self.project_root),
            "e2e_dir": str(self.e2e_dir),
            "artifacts_dir": str(self.artifacts_dir),
            "logs_dir": str(self.logs_dir),
            "ollama_base_url": self.ollama_base_url,
        }

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        return cls()

    def validate(self) -> None:
        """Validate configuration and raise ValidationError if invalid."""
        from .exceptions import ValidationError

        errors = []

        # Check required directories exist and are accessible
        if not self.e2e_dir.exists():
            errors.append(f"e2e directory does not exist: {self.e2e_dir}")

        if not self.policies_dir.exists():
            errors.append(f"policies directory does not exist: {self.policies_dir}")

        # Check OpenAI API key if using OpenAI models
        if self.model_provider in ["openai", "mixed"] and not self.openai_api_key:
            errors.append(
                "OPENAI_API_KEY environment variable is required for OpenAI model usage"
            )

        # Check MCP config exists
        if not self.mcp_config_path.exists():
            errors.append(f"MCP configuration file not found: {self.mcp_config_path}")

        if errors:
            raise ValidationError(
                "Configuration validation failed",
                validation_type="config",
                violations=errors,
            )
