"""
Configuration CLI utilities for QA Operator.

Provides command-line interface for configuration management,
validation, and health checks.
"""

import argparse
import json
import sys
from typing import Optional, Dict, Any
from pathlib import Path

from .config_manager import ConfigManager, get_config_manager
from .config_validator import validate_config, print_validation_results, ValidationLevel
from .config import Config


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate configuration command."""
    try:
        config_manager = get_config_manager()
        config = config_manager.get_config()
        
        print("🔍 Validating QA Operator configuration...")
        print()
        
        # Run validation
        results, has_errors = validate_config(config, quick=args.quick)
        
        # Print results
        print_validation_results(results, show_info=not args.quiet)
        
        # Return appropriate exit code
        return 1 if has_errors else 0
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return 1


def cmd_show(args: argparse.Namespace) -> int:
    """Show current configuration command."""
    try:
        config_manager = get_config_manager()
        config = config_manager.get_config()
        
        if args.format == "json":
            config_dict = config.to_dict()
            print(json.dumps(config_dict, indent=2))
        else:
            print("📋 Current QA Operator Configuration:")
            print()
            
            # Basic settings
            print("🔧 Basic Settings:")
            print(f"   CI Mode: {config.ci_mode}")
            print(f"   Headless Mode: {config.get_effective_headless_mode()}")
            print(f"   Log Level: {config.log_level}")
            print(f"   Log Format: {config.log_format}")
            print()
            
            # Model settings
            print("🤖 AI Model Settings:")
            print(f"   Provider: {config.model_provider}")
            print(f"   OpenAI API Key: {'✅ Set' if config.openai_api_key else '❌ Not set'}")
            print(f"   Ollama URL: {config.ollama_base_url}")
            print()
            
            # Directories
            print("📁 Directories:")
            print(f"   Project Root: {config.project_root}")
            print(f"   E2E Tests: {config.e2e_dir}")
            print(f"   Artifacts: {config.artifacts_dir}")
            print(f"   Logs: {config.logs_dir}")
            print(f"   Policies: {config.policies_dir}")
            print()
            
            # Artifact settings
            print("🗄️  Artifact Management:")
            print(f"   Retention Days: {config.artifact_retention_days}")
            print()
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to show configuration: {e}")
        return 1


def cmd_templates(args: argparse.Namespace) -> int:
    """List or apply configuration templates command."""
    try:
        config_manager = get_config_manager()
        
        if args.action == "list":
            templates = config_manager.list_templates()
            
            if not templates:
                print("📝 No configuration templates found")
                return 0
            
            print("📝 Available Configuration Templates:")
            print()
            
            for template_name in templates:
                try:
                    template = config_manager.get_template(template_name)
                    print(f"   • {template_name}")
                    print(f"     Description: {template.description}")
                    print(f"     Environment: {template.environment}")
                    if template.required_env_vars:
                        print(f"     Required Env Vars: {', '.join(template.required_env_vars)}")
                    print()
                except Exception as e:
                    print(f"   • {template_name} (Error: {e})")
                    print()
        
        elif args.action == "apply":
            if not args.template:
                print("❌ Template name required for apply action")
                return 1
            
            print(f"🔧 Applying configuration template: {args.template}")
            
            try:
                config = config_manager.apply_template(args.template, save=not args.dry_run)
                
                if args.dry_run:
                    print("🔍 Dry run - configuration would be:")
                    print(json.dumps(config.to_dict(), indent=2))
                else:
                    print("✅ Configuration template applied successfully")
                    
                    # Validate the new configuration
                    results, has_errors = validate_config(config, quick=True)
                    if has_errors:
                        print()
                        print("⚠️  Applied configuration has validation errors:")
                        print_validation_results(results, show_info=False)
                        return 1
                
            except ValueError as e:
                print(f"❌ Template error: {e}")
                return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ Template operation failed: {e}")
        return 1


def cmd_health(args: argparse.Namespace) -> int:
    """Health check command."""
    try:
        config_manager = get_config_manager()
        config = config_manager.get_config()
        
        print("🏥 QA Operator Health Check")
        print("=" * 40)
        print()
        
        # Configuration validation
        print("🔍 Configuration Validation:")
        results, has_errors = validate_config(config, quick=False)
        
        if has_errors:
            print("❌ Configuration has errors")
            error_results = [r for r in results if r.level == ValidationLevel.ERROR]
            for result in error_results[:3]:  # Show first 3 errors
                print(f"   • {result.message}")
            if len(error_results) > 3:
                print(f"   ... and {len(error_results) - 3} more errors")
        else:
            print("✅ Configuration is valid")
        
        print()
        
        # Directory checks
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
            warnings = [r for r in results if r.level == ValidationLevel.WARNING]
            if warnings:
                print("⚠️  Health check passed with warnings")
                print("   Run 'qa-operator config validate' for detailed information")
            else:
                print("✅ Health check passed - system ready")
            return 0
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return 1


def cmd_docs(args: argparse.Namespace) -> int:
    """Show configuration documentation command."""
    try:
        config_manager = get_config_manager()
        docs = config_manager.get_config_documentation()
        
        if args.format == "json":
            print(json.dumps(docs, indent=2))
        else:
            print("📚 QA Operator Configuration Documentation")
            print("=" * 50)
            print()
            
            # Environment variables
            print("🌍 Environment Variables:")
            print()
            for var_name, var_info in docs["environment_variables"].items():
                print(f"   {var_name}")
                print(f"     Description: {var_info['description']}")
                print(f"     Type: {var_info['type']}")
                print(f"     Default: {var_info['default']}")
                if 'options' in var_info:
                    print(f"     Options: {', '.join(var_info['options'])}")
                print(f"     Example: {var_info['example']}")
                print()
            
            # Configuration files
            print("📄 Configuration Files:")
            print()
            for file_name, file_info in docs["configuration_files"].items():
                print(f"   {file_name}")
                print(f"     Description: {file_info['description']}")
                print(f"     Location: {file_info['location']}")
                print(f"     Format: {file_info['format']}")
                print(f"     Hot Reload: {file_info['hot_reload']}")
                print()
            
            # Templates
            if docs["templates"]:
                print("📝 Configuration Templates:")
                print()
                for template_name, template_info in docs["templates"].items():
                    if "error" not in template_info:
                        print(f"   {template_name}")
                        print(f"     Description: {template_info['description']}")
                        print(f"     Environment: {template_info['environment']}")
                        print()
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to show documentation: {e}")
        return 1


def create_config_parser() -> argparse.ArgumentParser:
    """Create configuration CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="qa-operator config",
        description="QA Operator configuration management"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Configuration commands")
    
    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate configuration"
    )
    validate_parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation (skip network checks)"
    )
    validate_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only show errors and warnings"
    )
    validate_parser.set_defaults(func=cmd_validate)
    
    # Show command
    show_parser = subparsers.add_parser(
        "show",
        help="Show current configuration"
    )
    show_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format"
    )
    show_parser.set_defaults(func=cmd_show)
    
    # Templates command
    templates_parser = subparsers.add_parser(
        "templates",
        help="Manage configuration templates"
    )
    templates_parser.add_argument(
        "action",
        choices=["list", "apply"],
        help="Template action"
    )
    templates_parser.add_argument(
        "--template",
        help="Template name (for apply action)"
    )
    templates_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be applied without making changes"
    )
    templates_parser.set_defaults(func=cmd_templates)
    
    # Health command
    health_parser = subparsers.add_parser(
        "health",
        help="Run system health check"
    )
    health_parser.set_defaults(func=cmd_health)
    
    # Docs command
    docs_parser = subparsers.add_parser(
        "docs",
        help="Show configuration documentation"
    )
    docs_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format"
    )
    docs_parser.set_defaults(func=cmd_docs)
    
    return parser


def main(args: Optional[list] = None) -> int:
    """Main configuration CLI entry point."""
    parser = create_config_parser()
    
    if args is None:
        args = sys.argv[1:]
    
    parsed_args = parser.parse_args(args)
    
    if not hasattr(parsed_args, 'func'):
        parser.print_help()
        return 1
    
    return parsed_args.func(parsed_args)


if __name__ == "__main__":
    sys.exit(main())