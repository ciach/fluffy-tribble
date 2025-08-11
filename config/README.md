# QA Operator Configuration Guide

This guide covers all aspects of configuring the QA Operator for different environments and use cases.

## Quick Start

1. **Copy example configuration files:**
   ```bash
   cp config/examples/.env.example .env
   cp config/examples/qa-operator.config.example.json qa-operator.config.json
   ```

2. **Set required environment variables:**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

3. **Validate configuration:**
   ```bash
   python -m orchestrator.core.config_cli validate
   ```

## Configuration Files

### Main Configuration File

The main configuration file `qa-operator.config.json` contains all QA Operator settings:

```json
{
  "ci_mode": false,
  "headless_mode": null,
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
}
```

### Environment Variables

Environment variables override configuration file settings:

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `CI` | Enable CI mode (headless, JSON logs) | `false` | `true` |
| `QA_OPERATOR_HEADLESS` | Force headless mode | `null` | `true` |
| `QA_OPERATOR_LOG_LEVEL` | Set logging level | `INFO` | `DEBUG` |
| `QA_OPERATOR_MODEL_PROVIDER` | AI model provider | `mixed` | `openai` |
| `QA_OPERATOR_ARTIFACT_RETENTION_DAYS` | Artifact retention period | `7` (dev), `30` (CI) | `14` |
| `OPENAI_API_KEY` | OpenAI API key | - | `sk-...` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` | `http://localhost:11434` |

### MCP Configuration

The MCP configuration file `orchestrator/mcp.config.json` defines MCP servers:

```json
{
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
}
```

## Configuration Templates

Templates provide pre-configured settings for different environments:

### Available Templates

- **development**: Local development with debug logging
- **ci**: Continuous integration environment
- **production**: Production environment with minimal logging
- **local-ollama**: Local development using only Ollama models

### Using Templates

```bash
# List available templates
python -m orchestrator.core.config_cli templates list

# Apply a template
python -m orchestrator.core.config_cli templates apply --template development

# Preview template changes (dry run)
python -m orchestrator.core.config_cli templates apply --template ci --dry-run
```

## Environment-Specific Configuration

### Development Environment

```bash
# Apply development template
python -m orchestrator.core.config_cli templates apply --template development

# Or set environment variables
export QA_OPERATOR_LOG_LEVEL=DEBUG
export QA_OPERATOR_HEADLESS=false
```

**Characteristics:**
- Debug logging enabled
- Headed browser mode for visibility
- 7-day artifact retention
- Text log format

### CI/CD Environment

```bash
# Set CI environment variable
export CI=true
export OPENAI_API_KEY="your-api-key"

# Or apply CI template
python -m orchestrator.core.config_cli templates apply --template ci
```

**Characteristics:**
- Headless browser mode
- JSON log format for parsing
- 30-day artifact retention
- INFO level logging

### Production Environment

```bash
# Apply production template
python -m orchestrator.core.config_cli templates apply --template production

# Set required environment variables
export OPENAI_API_KEY="your-production-api-key"
```

**Characteristics:**
- Headless browser mode
- JSON log format
- WARN level logging
- 30-day artifact retention

## Configuration Management

### Validation

Validate your configuration before running:

```bash
# Full validation (includes network checks)
python -m orchestrator.core.config_cli validate

# Quick validation (local checks only)
python -m orchestrator.core.config_cli validate --quick

# Quiet mode (errors and warnings only)
python -m orchestrator.core.config_cli validate --quiet
```

### Health Check

Run a comprehensive health check:

```bash
python -m orchestrator.core.config_cli health
```

This checks:
- Configuration validity
- Directory permissions
- MCP server configuration
- AI model setup
- System requirements

### View Current Configuration

```bash
# Human-readable format
python -m orchestrator.core.config_cli show

# JSON format
python -m orchestrator.core.config_cli show --format json
```

### Hot Reloading

Configuration files are automatically reloaded when changed (if hot reloading is enabled in your application).

## AI Model Configuration

### Mixed Provider (Recommended)

Uses OpenAI for complex tasks and Ollama for cost-effective operations:

```bash
export QA_OPERATOR_MODEL_PROVIDER=mixed
export OPENAI_API_KEY="your-api-key"
export OLLAMA_BASE_URL="http://localhost:11434"
```

### OpenAI Only

Uses OpenAI for all operations:

```bash
export QA_OPERATOR_MODEL_PROVIDER=openai
export OPENAI_API_KEY="your-api-key"
```

### Ollama Only

Uses local Ollama for all operations (cost-effective but may be slower):

```bash
export QA_OPERATOR_MODEL_PROVIDER=ollama
export OLLAMA_BASE_URL="http://localhost:11434"

# Or apply template
python -m orchestrator.core.config_cli templates apply --template local-ollama
```

## Logging Configuration

### Log Levels

- **DEBUG**: Detailed information for debugging
- **INFO**: General information about operations
- **WARN**: Warning messages for potential issues
- **ERROR**: Error messages for failures

### Log Formats

- **text**: Human-readable format for development
- **json**: Structured format for CI/CD and production

### Log Files

- Main log: `logs/qa-operator.log`
- Debug logs: `logs/debug/` (when debug level enabled)

## Artifact Management

### Retention Policies

- **Development**: 7 days (default)
- **CI/CD**: 30 days (default)
- **Custom**: Set `QA_OPERATOR_ARTIFACT_RETENTION_DAYS`

### Artifact Types

- Test traces
- Screenshots
- Console logs
- Network logs
- Video recordings (if enabled)

### Storage Location

Artifacts are stored in `artifacts/{timestamp}/{test_name}/` for organized access.

## Troubleshooting

### Common Issues

1. **OpenAI API Key Missing**
   ```
   Error: OpenAI API key required for model provider 'mixed'
   Solution: Set OPENAI_API_KEY environment variable
   ```

2. **MCP Server Not Found**
   ```
   Error: Required command not found: npx
   Solution: Install Node.js and npm
   ```

3. **Directory Permissions**
   ```
   Error: Artifacts directory is not writable
   Solution: chmod 755 ./artifacts
   ```

4. **Invalid Configuration**
   ```
   Error: Invalid log level: INVALID
   Solution: Use DEBUG, INFO, WARN, or ERROR
   ```

### Getting Help

```bash
# Show configuration documentation
python -m orchestrator.core.config_cli docs

# Run health check for detailed diagnostics
python -m orchestrator.core.config_cli health

# Validate configuration with detailed output
python -m orchestrator.core.config_cli validate
```

## Advanced Configuration

### Custom Configuration File Location

```python
from orchestrator.core.config_manager import ConfigManager
from pathlib import Path

config_manager = ConfigManager(Path("/custom/path/config.json"))
```

### Programmatic Configuration

```python
from orchestrator.core.config import Config
from orchestrator.core.config_manager import get_config_manager

# Get current configuration
config = get_config_manager().get_config()

# Modify configuration
config.log_level = "DEBUG"

# Save changes
get_config_manager().save_config(config)
```

### Configuration Callbacks

```python
from orchestrator.core.config_manager import get_config_manager

def on_config_change(config):
    print(f"Configuration changed: {config.log_level}")

# Add callback for configuration changes
get_config_manager().add_reload_callback(on_config_change)

# Start hot reloading
get_config_manager().start_hot_reload()
```

## Security Considerations

1. **API Keys**: Never commit API keys to version control
2. **File Permissions**: Restrict access to configuration files
3. **Debug Logging**: Disable debug logging in production
4. **Environment Variables**: Use secure methods to set sensitive variables

## Best Practices

1. **Use Templates**: Start with appropriate environment template
2. **Validate Early**: Always validate configuration before deployment
3. **Monitor Logs**: Set appropriate log levels for your environment
4. **Artifact Cleanup**: Configure appropriate retention periods
5. **Version Control**: Track configuration changes (excluding secrets)