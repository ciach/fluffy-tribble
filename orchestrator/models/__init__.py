"""
AI model integration and routing system for QA Operator.

This module provides comprehensive model interaction capabilities including:
- Model routing between OpenAI and Ollama
- Prompt templating for different task types
- Response parsing and validation
- Context management for large inputs
- Rate limiting and retry logic
- High-level interaction utilities
"""

from .router import ModelRouter
from .types import TaskType, ModelProvider, ModelResponse
from .templates import PromptTemplateManager, PromptTemplate
from .parsers import ResponseParser, ParsedResponse
from .context import ContextManager, ContextStrategy
from .rate_limiter import ModelRateLimitManager, RateLimitConfig, RetryConfig
from .utilities import ModelInteractionManager

__all__ = [
    # Core types
    "TaskType",
    "ModelProvider",
    "ModelResponse",
    # Main components
    "ModelRouter",
    "PromptTemplateManager",
    "ResponseParser",
    "ContextManager",
    "ModelRateLimitManager",
    "ModelInteractionManager",
    # Supporting types
    "PromptTemplate",
    "ParsedResponse",
    "ContextStrategy",
    "RateLimitConfig",
    "RetryConfig",
]
