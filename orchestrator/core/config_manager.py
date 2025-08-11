"""
Configuration management system for QA Operator.

Provides configuration validation, defaults, templates, and hot-reloading
capabilities for all QA Operator components.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from dataclasses import dataclass, field, asdict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import time
from datetime import datetime

from .config import Config
from .exceptions import ValidationError


@dataclass
class ConfigTemplate:
    """Configuration template for different environments."""
    
    name: str
    description: str
    environment: str
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    required_env_vars: List[str] = field(default_factory=list)
    optional_env_vars: List[str] = field(default_factory=list)


class ConfigFileHandler(FileSystemEventHandler):
    """File system event handler for configuration hot-reloading."""
    
    def __init__(self, config_manager: 'ConfigManager'):
        self.config_manager = config_manager
        self.last_reload = time.time()
        self.reload_debounce = 1.0  # 1 second debounce
    
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
            
        # Check if it's a config file we care about
        config_files = [
            str(self.config_manager.config_file_path),
            str(self.config_manager.mcp_config_path),
            ".env"
        ]
        
        if any(event.src_path.endswith(cf) for cf in config_files):
            current_time = time.time()
            if current_time - self.last_reload > self.reload_debounce:
                self.last_reload = current_time
                self.config_manager._trigger_reload()


class ConfigManager:
    """
    Configuration management system with validation, templates, and hot-reloading.
    
    Features:
    - Configuration validation with detailed error reporting
    - Environment-specific configuration templates
    - Hot-reloading of configuration files
    - Configuration documentation and examples
    - Centralized configuration access
    """
    
    def __init__(self, config_file_path: Optional[Path] = None):
        self.config_file_path = config_file_path or Path.cwd() / "qa-operator.config.json"
        self.mcp_config_path = Path.cwd() / "orchestrator" / "mcp.config.json"
        self.templates_dir = Path.cwd() / "config" / "templates"
        self.examples_dir = Path.cwd() / "config" / "examples"
        
        self._config: Optional[Config] = None
        self._reload_callbacks: List[Callable[[Config], None]] = []
        self._observer: Optional[Observer] = None
        self._lock = threading.RLock()
        
        # Ensure directories exist
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.examples_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize templates
        self._create_default_templates()
        self._create_example_configs()
    
    def get_config(self) -> Config:
        """Get current configuration, loading if necessary."""
        with self._lock:
            if self._config is None:
                self._config = self._load_config()
            return self._config
    
    def reload_config(self) -> Config:
        """Force reload configuration from files."""
        with self._lock:
            self._config = self._load_config()
            self._notify_reload_callbacks()
            return self._config
    
    def validate_config(self, config: Optional[Config] = None) -> List[str]:
        """
        Validate configuration and return list of validation errors.
        
        Args:
            config: Configuration to validate. If None, uses current config.
            
        Returns:
            List of validation error messages. Empty list if valid.
        """
        if config is None:
            config = self.get_config()
        
        errors = []
        
        try:
            config.validate()
        except ValidationError as e:
            if hasattr(e, 'violations'):
                errors.extend(e.violations)
            else:
                errors.append(str(e))
        
        # Additional validation checks
        errors.extend(self._validate_mcp_config())
        errors.extend(self._validate_environment_consistency())
        
        return errors
    
    def get_template(self, template_name: str) -> ConfigTemplate:
        """Get configuration template by name."""
        template_path = self.templates_dir / f"{template_name}.json"
        if not template_path.exists():
            raise ValueError(f"Template '{template_name}' not found")
        
        with open(template_path, 'r') as f:
            template_data = json.load(f)
        
        return ConfigTemplate(**template_data)
    
    def list_templates(self) -> List[str]:
        """List available configuration templates."""
        return [f.stem for f in self.templates_dir.glob("*.json")]
    
    def apply_template(self, template_name: str, save: bool = True) -> Config:
        """
        Apply configuration template and optionally save to file.
        
        Args:
            template_name: Name of template to apply
            save: Whether to save the configuration to file
            
        Returns:
            New configuration with template applied
        """
        template = self.get_template(template_name)
        
        # Start with current config or defaults
        current_config = self.get_config()
        config_dict = asdict(current_config)
        
        # Apply template overrides
        config_dict.update(template.config_overrides)
        
        # Create new config instance
        new_config = Config(**config_dict)
        
        if save:
            self.save_config(new_config)
        
        with self._lock:
            self._config = new_config
            self._notify_reload_callbacks()
        
        return new_config
    
    def save_config(self, config: Config) -> None:
        """Save configuration to file."""
        config_dict = asdict(config)
        
        # Convert Path objects to strings for JSON serialization
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        
        # Remove sensitive information
        config_dict.pop('openai_api_key', None)
        
        with open(self.config_file_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    def start_hot_reload(self) -> None:
        """Start hot-reloading of configuration files."""
        if self._observer is not None:
            return  # Already started
        
        event_handler = ConfigFileHandler(self)
        self._observer = Observer()
        
        # Watch current directory for config files
        self._observer.schedule(event_handler, str(Path.cwd()), recursive=False)
        
        # Watch orchestrator directory for MCP config
        orchestrator_dir = Path.cwd() / "orchestrator"
        if orchestrator_dir.exists():
            self._observer.schedule(event_handler, str(orchestrator_dir), recursive=False)
        
        self._observer.start()
    
    def stop_hot_reload(self) -> None:
        """Stop hot-reloading of configuration files."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None
    
    def add_reload_callback(self, callback: Callable[[Config], None]) -> None:
        """Add callback to be called when configuration is reloaded."""
        self._reload_callbacks.append(callback)
    
    def remove_reload_callback(self, callback: Callable[[Config], None]) -> None:
        """Remove reload callback."""
        if callback in self._reload_callbacks:
            self._reload_callbacks.remove(callback)
    
    def get_config_documentation(self) -> Dict[str, Any]:
        """Get comprehensive configuration documentation."""
        return {
            "version": "1.0.0",
            "last_updated": datetime.now().isoformat(),
            "environment_variables": {
                "CI": {
                    "description": "Set to 'true' to enable CI mode (headless testing, JSON logging)",
                    "type": "boolean",
                    "default": "false",
                    "example": "true"
                },
                "QA_OPERATOR_HEADLESS": {
                    "description": "Override CI detection, force headless mode when 'true'",
                    "type": "boolean",
                    "default": "null",
                    "example": "true"
                },
                "QA_OPERATOR_LOG_LEVEL": {
                    "description": "Set logging level",
                    "type": "string",
                    "options": ["DEBUG", "INFO", "WARN", "ERROR"],
                    "default": "INFO",
                    "example": "DEBUG"
                },
                "QA_OPERATOR_MODEL_PROVIDER": {
                    "description": "Override default model routing",
                    "type": "string",
                    "options": ["openai", "ollama", "mixed"],
                    "default": "mixed",
                    "example": "openai"
                },
                "QA_OPERATOR_ARTIFACT_RETENTION_DAYS": {
                    "description": "Override default artifact retention period",
                    "type": "integer",
                    "default": "7 (local), 30 (CI)",
                    "example": "14"
                },
                "OPENAI_API_KEY": {
                    "description": "OpenAI API key for model access",
                    "type": "string",
                    "required_for": ["openai", "mixed"],
                    "example": "sk-..."
                },
                "OLLAMA_BASE_URL": {
                    "description": "Base URL for local Ollama instance",
                    "type": "string",
                    "default": "http://localhost:11434",
                    "example": "http://localhost:11434"
                }
            },
            "configuration_files": {
                "qa-operator.config.json": {
                    "description": "Main configuration file for QA Operator",
                    "location": str(self.config_file_path),
                    "format": "JSON",
                    "hot_reload": True
                },
                "orchestrator/mcp.config.json": {
                    "description": "MCP server configuration",
                    "location": str(self.mcp_config_path),
                    "format": "JSON",
                    "hot_reload": True
                }
            },
            "templates": {
                template: self._get_template_info(template)
                for template in self.list_templates()
            }
        }
    
    def _load_config(self) -> Config:
        """Load configuration from files and environment."""
        # Start with environment-based config
        config = Config.from_env()
        
        # Override with file-based config if it exists
        if self.config_file_path.exists():
            try:
                with open(self.config_file_path, 'r') as f:
                    file_config = json.load(f)
                
                # Apply file overrides to config
                for key, value in file_config.items():
                    if hasattr(config, key):
                        # Convert string paths back to Path objects
                        if key.endswith('_dir') or key.endswith('_path'):
                            value = Path(value)
                        setattr(config, key, value)
            except (json.JSONDecodeError, IOError) as e:
                # Log error but continue with environment config
                print(f"Warning: Could not load config file {self.config_file_path}: {e}")
        
        return config
    
    def _validate_mcp_config(self) -> List[str]:
        """Validate MCP configuration file."""
        errors = []
        
        if not self.mcp_config_path.exists():
            errors.append(f"MCP configuration file not found: {self.mcp_config_path}")
            return errors
        
        try:
            with open(self.mcp_config_path, 'r') as f:
                mcp_config = json.load(f)
            
            if 'mcpServers' not in mcp_config:
                errors.append("MCP configuration missing 'mcpServers' section")
            else:
                servers = mcp_config['mcpServers']
                required_servers = ['playwright', 'filesystem']
                
                for server in required_servers:
                    if server not in servers:
                        errors.append(f"Required MCP server '{server}' not configured")
                    else:
                        server_config = servers[server]
                        if 'command' not in server_config:
                            errors.append(f"MCP server '{server}' missing 'command' field")
                        if 'args' not in server_config:
                            errors.append(f"MCP server '{server}' missing 'args' field")
        
        except (json.JSONDecodeError, IOError) as e:
            errors.append(f"Invalid MCP configuration file: {e}")
        
        return errors
    
    def _validate_environment_consistency(self) -> List[str]:
        """Validate environment variable consistency."""
        errors = []
        
        # Check for conflicting environment variables
        ci_mode = os.getenv("CI", "").lower() == "true"
        headless_override = os.getenv("QA_OPERATOR_HEADLESS")
        
        if ci_mode and headless_override == "false":
            errors.append("Conflicting configuration: CI mode enabled but headless explicitly disabled")
        
        # Check model provider consistency
        model_provider = os.getenv("QA_OPERATOR_MODEL_PROVIDER", "mixed")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if model_provider in ["openai", "mixed"] and not openai_key:
            errors.append(f"Model provider '{model_provider}' requires OPENAI_API_KEY environment variable")
        
        return errors
    
    def _trigger_reload(self) -> None:
        """Trigger configuration reload (called by file watcher)."""
        try:
            self.reload_config()
        except Exception as e:
            print(f"Error reloading configuration: {e}")
    
    def _notify_reload_callbacks(self) -> None:
        """Notify all registered callbacks of configuration reload."""
        for callback in self._reload_callbacks:
            try:
                callback(self._config)
            except Exception as e:
                print(f"Error in reload callback: {e}")
    
    def _create_default_templates(self) -> None:
        """Create default configuration templates."""
        templates = [
            ConfigTemplate(
                name="development",
                description="Development environment configuration",
                environment="development",
                config_overrides={
                    "ci_mode": False,
                    "headless_mode": False,
                    "log_level": "DEBUG",
                    "log_format": "text",
                    "artifact_retention_days": 7
                },
                required_env_vars=[],
                optional_env_vars=["OPENAI_API_KEY", "OLLAMA_BASE_URL"]
            ),
            ConfigTemplate(
                name="ci",
                description="Continuous Integration environment configuration",
                environment="ci",
                config_overrides={
                    "ci_mode": True,
                    "headless_mode": True,
                    "log_level": "INFO",
                    "log_format": "json",
                    "artifact_retention_days": 30
                },
                required_env_vars=["OPENAI_API_KEY"],
                optional_env_vars=["QA_OPERATOR_LOG_LEVEL"]
            ),
            ConfigTemplate(
                name="production",
                description="Production environment configuration",
                environment="production",
                config_overrides={
                    "ci_mode": True,
                    "headless_mode": True,
                    "log_level": "WARN",
                    "log_format": "json",
                    "artifact_retention_days": 30
                },
                required_env_vars=["OPENAI_API_KEY"],
                optional_env_vars=[]
            ),
            ConfigTemplate(
                name="local-ollama",
                description="Local development with Ollama-only configuration",
                environment="development",
                config_overrides={
                    "ci_mode": False,
                    "headless_mode": False,
                    "log_level": "DEBUG",
                    "model_provider": "ollama",
                    "artifact_retention_days": 7
                },
                required_env_vars=[],
                optional_env_vars=["OLLAMA_BASE_URL"]
            )
        ]
        
        for template in templates:
            template_path = self.templates_dir / f"{template.name}.json"
            if not template_path.exists():
                with open(template_path, 'w') as f:
                    json.dump(asdict(template), f, indent=2)
    
    def _create_example_configs(self) -> None:
        """Create example configuration files."""
        examples = {
            "qa-operator.config.example.json": {
                "ci_mode": False,
                "headless_mode": None,
                "log_level": "INFO",
                "log_format": "text",
                "model_provider": "mixed",
                "artifact_retention_days": 7,
                "project_root": ".",
                "e2e_dir": "./e2e",
                "artifacts_dir": "./artifacts",
                "logs_dir": "./logs",
                "policies_dir": "./policies",
                "ollama_base_url": "http://localhost:11434"
            },
            "mcp.config.example.json": {
                "mcpServers": {
                    "playwright": {
                        "command": "npx",
                        "args": ["@playwright/mcp@latest"]
                    },
                    "filesystem": {
                        "command": "node",
                        "args": ["./servers/fs-mcp.js"]
                    },
                    "git": {
                        "command": "uvx",
                        "args": ["git-mcp-server@latest"]
                    }
                }
            },
            ".env.example": [
                "# QA Operator Environment Configuration",
                "",
                "# CI/CD Environment",
                "CI=false",
                "",
                "# Execution Mode",
                "QA_OPERATOR_HEADLESS=false",
                "",
                "# Logging",
                "QA_OPERATOR_LOG_LEVEL=INFO",
                "",
                "# AI Model Configuration",
                "QA_OPERATOR_MODEL_PROVIDER=mixed",
                "OPENAI_API_KEY=your-openai-api-key-here",
                "OLLAMA_BASE_URL=http://localhost:11434",
                "",
                "# Artifact Management",
                "QA_OPERATOR_ARTIFACT_RETENTION_DAYS=7",
                ""
            ]
        }
        
        for filename, content in examples.items():
            example_path = self.examples_dir / filename
            if not example_path.exists():
                if isinstance(content, list):
                    # Handle .env file as list of lines
                    with open(example_path, 'w') as f:
                        f.write('\n'.join(content))
                else:
                    # Handle JSON files
                    with open(example_path, 'w') as f:
                        json.dump(content, f, indent=2)
    
    def _get_template_info(self, template_name: str) -> Dict[str, Any]:
        """Get information about a configuration template."""
        try:
            template = self.get_template(template_name)
            return {
                "description": template.description,
                "environment": template.environment,
                "required_env_vars": template.required_env_vars,
                "optional_env_vars": template.optional_env_vars,
                "config_overrides": list(template.config_overrides.keys())
            }
        except Exception:
            return {"error": "Template not found or invalid"}


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> Config:
    """Get current configuration from global manager."""
    return get_config_manager().get_config()