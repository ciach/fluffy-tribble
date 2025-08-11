"""
Main CLI interface for QA Operator.

Provides command-line interface for manual execution, configuration validation,
health checks, and diagnostic utilities.
"""

import argparse
import sys
import os
import json
from typing import Optional, List
from pathlib import Path

from .core.config_manager import get_config_manager
from .core.config_cli import create_config_parser, main as config_main
from .core.config_validator import validate_config, print_validation_results
from .agent import QAOperatorAgent
from .core.exceptions import QAOperatorError


def cmd_run(args: argparse.Namespace) -> int:
    """Run QA Operator workflow command."""
    try:
        print("🚀 Starting QA Operator workflow...")
        
        # Get configuration
        config_manager = get_config_manager()
        config = config_manager.get_config()
        
        # Validate configuration first
        print("🔍 Validating configuration...")
        results, has_errors = validate_config(config, quick=True)
        
        if has_errors:
            print("❌ Configuration validation failed:")
            print_validation_results(results, show_info=False)
            return 1
        
        # Initialize agent
        print("🤖 Initializing QA Operator agent...")
        agent = QAOperatorAgent(config)
        
        # Prepare specification
        if args.spec_file:
            spec_path = Path(args.spec_file)
            if not spec_path.exists():
                print(f"❌ Specification file not found: {spec_path}")
                return 1
            
            with open(spec_path, 'r') as f:
                if spec_path.suffix.lower() == '.json':
                    specification = json.load(f)
                else:
                    specification = f.read()
        else:
            # Interactive mode or default specification
            specification = {
                "name": "default",
                "description": "Default QA Operator workflow",
                "requirements": []
            }
        
        # Run workflow
        print("⚙️  Running QA Operator workflow...")
        result = agent.run_workflow(specification)
        
        if result.get('success', False):
            print("✅ QA Operator workflow completed successfully!")
            if args.verbose:
                print(f"📊 Results: {json.dumps(result, indent=2)}")
            return 0
        else:
            print("❌ QA Operator workflow failed!")
            if 'error' in result:
                print(f"Error: {result['error']}")
            return 1
            
    except QAOperatorError as e:
        print(f"❌ QA Operator error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_health(args: argparse.Namespace) -> int:
    """Health check command."""
    try:
        print("🏥 QA Operator Health Check")
        print("=" * 40)
        print()
        
        config_manager = get_config_manager()
        config = config_manager.get_config()
        
        # Configuration validation
        print("🔍 Configuration Validation:")
        results, has_errors = validate_config(config, quick=False)
        
        if has_errors:
            print("❌ Configuration has errors")
            error_results = [r for r in results if "error" in r.message.lower()]
            for result in error_results[:3]:  # Show first 3 errors
                print(f"   • {result.message}")
            if len(error_results) > 3:
                print(f"   ... and {len(error_results) - 3} more errors")
        else:
            print("✅ Configuration is valid")
        
        print()
        
        # System requirements
        print("🔧 System Requirements:")
        
        # Python version
        python_version = sys.version_info
        if python_version >= (3, 8):
            print(f"   ✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        else:
            print(f"   ❌ Python {python_version.major}.{python_version.minor}.{python_version.micro} (requires 3.8+)")
        
        # Check required commands
        required_commands = ["node", "npm"]
        for cmd in required_commands:
            if _command_exists(cmd):
                print(f"   ✅ {cmd} available")
            else:
                print(f"   ❌ {cmd} not found")
        
        print()
        
        # Directory status
        print("📁 Directory Status:")
        directories = [
            ("E2E Tests", config.e2e_dir),
            ("Artifacts", config.artifacts_dir),
            ("Logs", config.logs_dir),
            ("Policies", config.policies_dir)
        ]
        
        for name, path in directories:
            if path.exists():
                print(f"   ✅ {name}: {path}")
            else:
                print(f"   ❌ {name}: {path} (missing)")
        
        print()
        
        # MCP Configuration
        print("🔌 MCP Configuration:")
        if config.mcp_config_path.exists():
            try:
                with open(config.mcp_config_path, 'r') as f:
                    mcp_config = json.load(f)
                
                servers = mcp_config.get('mcpServers', {})
                for server_name in ['playwright', 'filesystem', 'git']:
                    if server_name in servers:
                        print(f"   ✅ {server_name.title()} server configured")
                    else:
                        status = "❌" if server_name in ['playwright', 'filesystem'] else "⚠️ "
                        print(f"   {status} {server_name.title()} server not configured")
            except Exception as e:
                print(f"   ❌ MCP config error: {e}")
        else:
            print(f"   ❌ MCP config file not found: {config.mcp_config_path}")
        
        print()
        
        # Model Configuration
        print("🤖 AI Model Configuration:")
        if config.model_provider in ["openai", "mixed"]:
            if config.openai_api_key:
                print("   ✅ OpenAI API key configured")
            else:
                print("   ❌ OpenAI API key missing")
        
        if config.model_provider in ["ollama", "mixed"]:
            print(f"   ℹ️  Ollama URL: {config.ollama_base_url}")
        
        print()
        
        # Summary
        if has_errors:
            print("❌ Health check failed - configuration errors found")
            print("   Run 'qa-operator config validate' for detailed information")
            return 1
        else:
            warnings = [r for r in results if "warning" in r.message.lower()]
            if warnings:
                print("⚠️  Health check passed with warnings")
                print("   Run 'qa-operator config validate' for detailed information")
            else:
                print("✅ Health check passed - system ready")
            return 0
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return 1


def cmd_version(args: argparse.Namespace) -> int:
    """Show version information."""
    try:
        # Try to get version from package metadata
        try:
            import pkg_resources
            version = pkg_resources.get_distribution("qa-operator").version
        except:
            version = "development"
        
        print(f"QA Operator {version}")
        
        if args.verbose:
            print()
            print("System Information:")
            print(f"  Python: {sys.version}")
            print(f"  Platform: {sys.platform}")
            print(f"  Working Directory: {os.getcwd()}")
            
            # Configuration info
            try:
                config = get_config_manager().get_config()
                print(f"  Configuration: {config.project_root}")
                print(f"  Model Provider: {config.model_provider}")
                print(f"  Log Level: {config.log_level}")
            except Exception:
                print("  Configuration: Not available")
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to get version information: {e}")
        return 1


def cmd_init(args: argparse.Namespace) -> int:
    """Initialize QA Operator in current directory."""
    try:
        print("🚀 Initializing QA Operator...")
        
        current_dir = Path.cwd()
        
        # Create required directories
        directories = ["e2e", "artifacts", "logs", "policies", "config/examples"]
        for dir_name in directories:
            dir_path = current_dir / dir_name
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"   ✅ Created directory: {dir_path}")
            else:
                print(f"   ℹ️  Directory already exists: {dir_path}")
        
        # Copy example configuration files
        config_manager = get_config_manager()
        
        # Create example files if they don't exist
        example_files = {
            ".env.example": """# QA Operator Environment Configuration

# CI/CD Environment
CI=false

# Execution Mode
QA_OPERATOR_HEADLESS=false

# Logging
QA_OPERATOR_LOG_LEVEL=INFO

# AI Model Configuration
QA_OPERATOR_MODEL_PROVIDER=mixed
OPENAI_API_KEY=your-openai-api-key-here
OLLAMA_BASE_URL=http://localhost:11434

# Artifact Management
QA_OPERATOR_ARTIFACT_RETENTION_DAYS=7
""",
            "qa-operator.config.example.json": json.dumps({
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
            }, indent=2),
            "orchestrator/mcp.config.json": json.dumps({
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
            }, indent=2)
        }
        
        for filename, content in example_files.items():
            file_path = current_dir / filename
            if not file_path.exists():
                # Create parent directory if needed
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, 'w') as f:
                    f.write(content)
                print(f"   ✅ Created file: {file_path}")
            else:
                print(f"   ℹ️  File already exists: {file_path}")
        
        # Create basic selector policy
        selector_policy_path = current_dir / "policies" / "selector.md"
        if not selector_policy_path.exists():
            selector_policy_content = """# Selector Policy

## Preferred Selectors (in order of preference)

1. **getByRole()** - Most semantic and accessible
2. **getByLabel()** - Good for form elements
3. **getByTestId()** - Stable for testing
4. **getByText()** - For unique text content
5. **CSS selectors** - Last resort only

## Guidelines

- Always prefer semantic selectors over CSS selectors
- Use data-testid attributes for complex components
- Avoid brittle selectors like nth-child or complex CSS paths
- Include justifying comments for any CSS selectors used

## Examples

```typescript
// ✅ Good
await page.getByRole('button', { name: 'Submit' }).click();
await page.getByLabel('Email').fill('user@example.com');
await page.getByTestId('login-form').isVisible();

// ❌ Avoid
await page.locator('.btn-primary:nth-child(2)').click();
await page.locator('#form > div > input[type="email"]').fill('user@example.com');
```
"""
            with open(selector_policy_path, 'w') as f:
                f.write(selector_policy_content)
            print(f"   ✅ Created selector policy: {selector_policy_path}")
        
        print()
        print("✅ QA Operator initialization complete!")
        print()
        print("Next steps:")
        print("1. Copy .env.example to .env and configure your API keys")
        print("2. Run 'qa-operator health' to verify your setup")
        print("3. Run 'qa-operator config validate' to check configuration")
        print("4. Start using QA Operator with 'qa-operator run'")
        
        return 0
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return 1


def _command_exists(command: str) -> bool:
    """Check if a command exists in the system PATH."""
    import subprocess
    try:
        subprocess.run([command, "--version"], 
                     capture_output=True, 
                     check=False, 
                     timeout=5)
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def create_main_parser() -> argparse.ArgumentParser:
    """Create main CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="qa-operator",
        description="QA Operator - Intelligent end-to-end testing automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  qa-operator run --spec-file tests/login.json
  qa-operator health
  qa-operator config validate
  qa-operator init
  qa-operator version --verbose

For more information, visit: https://github.com/your-org/qa-operator
        """
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run QA Operator workflow"
    )
    run_parser.add_argument(
        "--spec-file",
        help="Path to test specification file (JSON or text)"
    )
    run_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    run_parser.set_defaults(func=cmd_run)
    
    # Health command
    health_parser = subparsers.add_parser(
        "health",
        help="Run system health check"
    )
    health_parser.set_defaults(func=cmd_health)
    
    # Version command
    version_parser = subparsers.add_parser(
        "version",
        help="Show version information"
    )
    version_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed version information"
    )
    version_parser.set_defaults(func=cmd_version)
    
    # Init command
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize QA Operator in current directory"
    )
    init_parser.set_defaults(func=cmd_init)
    
    # Config command (delegate to config CLI)
    config_parser = subparsers.add_parser(
        "config",
        help="Configuration management"
    )
    config_parser.set_defaults(func=lambda args: config_main(sys.argv[2:]))
    
    return parser


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = create_main_parser()
    
    if args is None:
        args = sys.argv[1:]
    
    # Handle special case for config command
    if len(args) > 0 and args[0] == "config":
        return config_main(args[1:])
    
    parsed_args = parser.parse_args(args)
    
    if not hasattr(parsed_args, 'func'):
        parser.print_help()
        return 1
    
    return parsed_args.func(parsed_args)


if __name__ == "__main__":
    sys.exit(main())