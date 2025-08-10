"""
Test execution components for QA Operator.

This module provides test execution, artifact management, and result processing
functionality for Playwright end-to-end tests.
"""

from .executor import TestExecutor
from .artifacts import ArtifactManager
from .models import (
    ExecutionConfig,
    ExecutionResult,
    TestExecutionError,
    ArtifactMetadata,
)

__all__ = [
    "TestExecutor",
    "ArtifactManager",
    "ExecutionConfig",
    "ExecutionResult",
    "TestExecutionError",
    "ArtifactMetadata",
]
