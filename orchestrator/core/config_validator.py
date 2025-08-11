"""
Configuration validation utilities for QA Operator.

Provides comprehensive validation of configuration settings,
environment variables, and system requirements.
"""

import os
import json
import subprocess
import sys
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from .config import Config
from .exceptions import ValidationError


class ValidationLevel(Enum):
    """Validation severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of a configuration validation check."""
    
    level: ValidationLevel
    category: str
    message: str
    suggestion: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class ConfigValidator:
    """
    Comprehensive configuration validator for QA Operator.
    
    Validates:
    - Configuration settings and consistency
    - Environment variables and their values
    - System requirements and dependencies
    - File permissions and accessibility
    - MCP server configurations
    - AI model connectivity
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.results: List[ValidationResult] = []
    
    def validate_all(self) -> List[ValidationResult]:
        """Run all validation checks and return results."""
        self.results = []
        
        # Core configuration validation
        self._validate_basic_config()
        self._validate_environment_variables()
        self._validate_directories()
        self._validate_file_permissions()
        
        # System requirements validation
        self._validate_system_requirements()
        self._validate_python_dependencies()
        
        # MCP and external service validation
        self._validate_mcp_configuration()
        self._validate_ai_model_connectivity()
        
        # Security and best practices validation
        self._validate_security_settings()
        self._validate_best_practices()
        
        return self.results
    
    def validate_quick(self) -> List[ValidationResult]:
        """Run quick validation checks (no network calls)."""
        self.results = []
        
        self._validate_basic_config()
        self._validate_environment_variables()
        self._validate_directories()
        
        return self.results
    
    def get_errors(self) -> List[ValidationResult]:
        """Get only error-level validation results."""
        return [r for r in self.results if r.level == ValidationLevel.ERROR]
    
    def get_warnings(self) -> List[ValidationResult]:
        """Get only warning-level validation results."""
        return [r for r in self.results if r.level == ValidationLevel.WARNING]
    
    def has_errors(self) -> bool:
        """Check if validation found any errors."""
        return len(self.get_errors()) > 0
    
    def _add_result(self, level: ValidationLevel, category: str, message: str, 
                   suggestion: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Add a validation result."""
        self.results.append(ValidationResult(
            level=level,
            category=category,
            message=message,
            suggestion=suggestion,
            details=details
        ))
    
    def _validate_basic_config(self):
        """Validate basic configuration settings."""
        # Log level validation
        valid_log_levels = ["DEBUG", "INFO", "WARN", "ERROR"]
        if self.config.log_level not in valid_log_levels:
            self._add_result(
                ValidationLevel.ERROR,
                "config",
                f"Invalid log level: {self.config.log_level}",
                f"Must be one of: {', '.join(valid_log_levels)}"
            )
        
        # Model provider validation
        valid_providers = ["openai", "ollama", "mixed"]
        if self.config.model_provider not in valid_providers:
            self._add_result(
                ValidationLevel.ERROR,
                "config",
                f"Invalid model provider: {self.config.model_provider}",
                f"Must be one of: {', '.join(valid_providers)}"
            )
        
        # Artifact retention validation
        if self.config.artifact_retention_days < 1:
            self._add_result(
                ValidationLevel.ERROR,
                "config",
                f"Invalid artifact retention days: {self.config.artifact_retention_days}",
                "Must be at least 1 day"
            )
        elif self.config.artifact_retention_days > 365:
            self._add_result(
                ValidationLevel.WARNING,
                "config",
                f"Very long artifact retention: {self.config.artifact_retention_days} days",
                "Consider shorter retention period to save disk space"
            )
        
        # CI mode consistency
        if self.config.ci_mode and self.config.headless_mode is False:
            self._add_result(
                ValidationLevel.WARNING,
                "config",
                "CI mode enabled but headless mode disabled",
                "Consider enabling headless mode for CI environments"
            )
    
    def _validate_environment_variables(self):
        """Validate environment variables and their consistency."""
        # OpenAI API key validation
        if self.config.model_provider in ["openai", "mixed"]:
            if not self.config.openai_api_key:
                self._add_result(
                    ValidationLevel.ERROR,
                    "environment",
                    f"OpenAI API key required for model provider '{self.config.model_provider}'",
                    "Set OPENAI_API_KEY environment variable"
                )
            elif not self.config.openai_api_key.startswith("sk-"):
                self._add_result(
                    ValidationLevel.WARNING,
                    "environment",
                    "OpenAI API key format appears invalid",
                    "OpenAI API keys typically start with 'sk-'"
                )
        
        # Ollama URL validation
        if self.config.model_provider in ["ollama", "mixed"]:
            if not self.config.ollama_base_url:
                self._add_result(
                    ValidationLevel.ERROR,
                    "environment",
                    "Ollama base URL required for Ollama model usage",
                    "Set OLLAMA_BASE_URL environment variable"
                )
            elif not self.config.ollama_base_url.startswith(("http://", "https://")):
                self._add_result(
                    ValidationLevel.ERROR,
                    "environment",
                    f"Invalid Ollama URL format: {self.config.ollama_base_url}",
                    "URL must start with http:// or https://"
                )
        
        # Environment variable conflicts
        ci_env = os.getenv("CI", "").lower() == "true"
        headless_env = os.getenv("QA_OPERATOR_HEADLESS", "").lower()
        
        if ci_env and headless_env == "false":
            self._add_result(
                ValidationLevel.WARNING,
                "environment",
                "Conflicting environment variables: CI=true but QA_OPERATOR_HEADLESS=false",
                "Consider removing QA_OPERATOR_HEADLESS override in CI"
            )
    
    def _validate_directories(self):
        """Validate directory existence and permissions."""
        directories = [
            ("e2e_dir", self.config.e2e_dir, "E2E tests directory"),
            ("artifacts_dir", self.config.artifacts_dir, "Artifacts directory"),
            ("logs_dir", self.config.logs_dir, "Logs directory"),
            ("policies_dir", self.config.policies_dir, "Policies directory")
        ]
        
        for dir_name, dir_path, description in directories:
            if not dir_path.exists():
                if dir_name in ["artifacts_dir", "logs_dir"]:
                    # These can be created automatically
                    self._add_result(
                        ValidationLevel.INFO,
                        "directories",
                        f"{description} will be created: {dir_path}",
                        "Directory will be created automatically on first use"
                    )
                else:
                    # These should exist
                    self._add_result(
                        ValidationLevel.ERROR,
                        "directories",
                        f"{description} does not exist: {dir_path}",
                        f"Create the directory: mkdir -p {dir_path}"
                    )
            elif not dir_path.is_dir():
                self._add_result(
                    ValidationLevel.ERROR,
                    "directories",
                    f"{description} is not a directory: {dir_path}",
                    f"Remove the file and create directory: rm {dir_path} && mkdir -p {dir_path}"
                )
    
    def _validate_file_permissions(self):
        """Validate file and directory permissions."""
        # Check write permissions for key directories
        writable_dirs = [
            (self.config.artifacts_dir, "Artifacts directory"),
            (self.config.logs_dir, "Logs directory")
        ]
        
        for dir_path, description in writable_dirs:
            if dir_path.exists():
                if not os.access(dir_path, os.W_OK):
                    self._add_result(
                        ValidationLevel.ERROR,
                        "permissions",
                        f"{description} is not writable: {dir_path}",
                        f"Fix permissions: chmod 755 {dir_path}"
                    )
        
        # Check MCP config file
        if self.config.mcp_config_path.exists():
            if not os.access(self.config.mcp_config_path, os.R_OK):
                self._add_result(
                    ValidationLevel.ERROR,
                    "permissions",
                    f"MCP config file is not readable: {self.config.mcp_config_path}",
                    f"Fix permissions: chmod 644 {self.config.mcp_config_path}"
                )
    
    def _validate_system_requirements(self):
        """Validate system requirements and dependencies."""
        # Python version check
        python_version = sys.version_info
        if python_version < (3, 8):
            self._add_result(
                ValidationLevel.ERROR,
                "system",
                f"Python version {python_version.major}.{python_version.minor} is too old",
                "QA Operator requires Python 3.8 or newer"
            )
        elif python_version < (3, 10):
            self._add_result(
                ValidationLevel.WARNING,
                "system",
                f"Python version {python_version.major}.{python_version.minor} is supported but not optimal",
                "Consider upgrading to Python 3.10+ for best performance"
            )
        
        # Check for required system commands
        required_commands = ["node", "npm"]
        for cmd in required_commands:
            if not self._command_exists(cmd):
                self._add_result(
                    ValidationLevel.ERROR,
                    "system",
                    f"Required command not found: {cmd}",
                    f"Install {cmd} to use Playwright MCP server"
                )
        
        # Check for optional commands
        optional_commands = {"uvx": "Git MCP server", "git": "Version control"}
        for cmd, purpose in optional_commands.items():
            if not self._command_exists(cmd):
                self._add_result(
                    ValidationLevel.INFO,
                    "system",
                    f"Optional command not found: {cmd}",
                    f"Install {cmd} to enable {purpose}"
                )
    
    def _validate_python_dependencies(self):
        """Validate Python package dependencies."""
        required_packages = [
            "playwright",
            "openai",
            "litellm",
            "watchdog",
            "yaml"  # pyyaml imports as yaml
        ]
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                self._add_result(
                    ValidationLevel.ERROR,
                    "dependencies",
                    f"Required Python package not found: {package}",
                    f"Install with: pip install {package}"
                )
    
    def _validate_mcp_configuration(self):
        """Validate MCP server configuration."""
        if not self.config.mcp_config_path.exists():
            self._add_result(
                ValidationLevel.ERROR,
                "mcp",
                f"MCP configuration file not found: {self.config.mcp_config_path}",
                "Create MCP configuration file with required servers"
            )
            return
        
        try:
            with open(self.config.mcp_config_path, 'r') as f:
                mcp_config = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self._add_result(
                ValidationLevel.ERROR,
                "mcp",
                f"Invalid MCP configuration file: {e}",
                "Fix JSON syntax in MCP configuration file"
            )
            return
        
        # Validate MCP structure
        if 'mcpServers' not in mcp_config:
            self._add_result(
                ValidationLevel.ERROR,
                "mcp",
                "MCP configuration missing 'mcpServers' section",
                "Add 'mcpServers' object to configuration"
            )
            return
        
        servers = mcp_config['mcpServers']
        required_servers = ['playwright', 'filesystem']
        optional_servers = ['git']
        
        # Check required servers
        for server in required_servers:
            if server not in servers:
                self._add_result(
                    ValidationLevel.ERROR,
                    "mcp",
                    f"Required MCP server '{server}' not configured",
                    f"Add {server} server configuration"
                )
            else:
                self._validate_mcp_server_config(server, servers[server])
        
        # Check optional servers
        for server in optional_servers:
            if server not in servers:
                self._add_result(
                    ValidationLevel.INFO,
                    "mcp",
                    f"Optional MCP server '{server}' not configured",
                    f"Add {server} server for additional functionality"
                )
            else:
                self._validate_mcp_server_config(server, servers[server])
    
    def _validate_mcp_server_config(self, server_name: str, server_config: Dict[str, Any]):
        """Validate individual MCP server configuration."""
        required_fields = ['command', 'args']
        
        for field in required_fields:
            if field not in server_config:
                self._add_result(
                    ValidationLevel.ERROR,
                    "mcp",
                    f"MCP server '{server_name}' missing '{field}' field",
                    f"Add '{field}' to {server_name} server configuration"
                )
        
        # Validate command exists
        if 'command' in server_config:
            command = server_config['command']
            if not self._command_exists(command):
                self._add_result(
                    ValidationLevel.WARNING,
                    "mcp",
                    f"MCP server '{server_name}' command not found: {command}",
                    f"Install {command} to use {server_name} server"
                )
    
    def _validate_ai_model_connectivity(self):
        """Validate AI model connectivity (network calls)."""
        # This is a placeholder for network-based validation
        # In a real implementation, you might test API connectivity
        if self.config.model_provider in ["openai", "mixed"] and self.config.openai_api_key:
            self._add_result(
                ValidationLevel.INFO,
                "models",
                "OpenAI API key configured",
                "Run health check to verify connectivity"
            )
        
        if self.config.model_provider in ["ollama", "mixed"]:
            self._add_result(
                ValidationLevel.INFO,
                "models",
                f"Ollama configured at {self.config.ollama_base_url}",
                "Run health check to verify Ollama server is running"
            )
    
    def _validate_security_settings(self):
        """Validate security-related settings."""
        # Check for sensitive data in logs
        if self.config.log_level == "DEBUG":
            self._add_result(
                ValidationLevel.WARNING,
                "security",
                "Debug logging enabled - may expose sensitive information",
                "Use INFO or higher log level in production"
            )
        
        # Check file permissions on config files
        config_files = [self.config.mcp_config_path]
        for config_file in config_files:
            if config_file.exists():
                stat = config_file.stat()
                # Check if file is world-readable (octal 004)
                if stat.st_mode & 0o004:
                    self._add_result(
                        ValidationLevel.WARNING,
                        "security",
                        f"Configuration file is world-readable: {config_file}",
                        f"Restrict permissions: chmod 600 {config_file}"
                    )
    
    def _validate_best_practices(self):
        """Validate configuration against best practices."""
        # Artifact retention best practices
        if self.config.ci_mode and self.config.artifact_retention_days < 30:
            self._add_result(
                ValidationLevel.INFO,
                "best_practices",
                f"Short artifact retention in CI: {self.config.artifact_retention_days} days",
                "Consider 30+ days retention for CI environments"
            )
        
        if not self.config.ci_mode and self.config.artifact_retention_days > 14:
            self._add_result(
                ValidationLevel.INFO,
                "best_practices",
                f"Long artifact retention in development: {self.config.artifact_retention_days} days",
                "Consider shorter retention for development environments"
            )
        
        # Model provider best practices
        if self.config.model_provider == "openai":
            self._add_result(
                ValidationLevel.INFO,
                "best_practices",
                "Using OpenAI-only model provider",
                "Consider 'mixed' provider to reduce costs with local models"
            )
    
    def _command_exists(self, command: str) -> bool:
        """Check if a command exists in the system PATH."""
        try:
            subprocess.run([command, "--version"], 
                         capture_output=True, 
                         check=False, 
                         timeout=5)
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False


def validate_config(config: Config, quick: bool = False) -> Tuple[List[ValidationResult], bool]:
    """
    Validate configuration and return results.
    
    Args:
        config: Configuration to validate
        quick: If True, skip network-based validation checks
        
    Returns:
        Tuple of (validation_results, has_errors)
    """
    validator = ConfigValidator(config)
    
    if quick:
        results = validator.validate_quick()
    else:
        results = validator.validate_all()
    
    has_errors = validator.has_errors()
    
    return results, has_errors


def print_validation_results(results: List[ValidationResult], show_info: bool = True):
    """Print validation results in a human-readable format."""
    if not results:
        print("✅ Configuration validation passed - no issues found")
        return
    
    # Group results by level
    errors = [r for r in results if r.level == ValidationLevel.ERROR]
    warnings = [r for r in results if r.level == ValidationLevel.WARNING]
    info = [r for r in results if r.level == ValidationLevel.INFO]
    
    # Print errors
    if errors:
        print(f"❌ {len(errors)} Error(s) found:")
        for result in errors:
            print(f"   • {result.message}")
            if result.suggestion:
                print(f"     → {result.suggestion}")
        print()
    
    # Print warnings
    if warnings:
        print(f"⚠️  {len(warnings)} Warning(s) found:")
        for result in warnings:
            print(f"   • {result.message}")
            if result.suggestion:
                print(f"     → {result.suggestion}")
        print()
    
    # Print info (if requested)
    if info and show_info:
        print(f"ℹ️  {len(info)} Info message(s):")
        for result in info:
            print(f"   • {result.message}")
            if result.suggestion:
                print(f"     → {result.suggestion}")
        print()
    
    # Summary
    if errors:
        print("❌ Configuration validation failed - please fix errors before proceeding")
    elif warnings:
        print("⚠️  Configuration validation passed with warnings")
    else:
        print("✅ Configuration validation passed")