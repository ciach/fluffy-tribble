"""
QA Operator - Intelligent Playwright Testing Agent

An automated testing agent that generates, executes, and maintains Playwright tests
using AI models and MCP (Model Context Protocol) integration.
"""

__version__ = "0.1.0"
__author__ = "QA Operator Team"

from .core.config import Config
from .core.exceptions import QAOperatorError
from .core.logging_config import setup_logging
from .core.workflow import WorkflowManager

__all__ = [
    "Config",
    "QAOperatorError",
    "setup_logging",
    "WorkflowManager",
]
