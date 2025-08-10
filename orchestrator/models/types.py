"""
Type definitions for the model integration system.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime


class TaskType(Enum):
    """Types of tasks that can be routed to different models."""

    PLANNING = "planning"
    DEBUGGING = "debugging"
    DRAFTING = "drafting"
    ANALYSIS = "analysis"
    GENERATION = "generation"


class ModelProvider(Enum):
    """Available model providers."""

    OPENAI = "openai"
    OLLAMA = "ollama"


@dataclass
class ModelResponse:
    """Response from a model interaction."""

    content: str
    provider: ModelProvider
    model_name: str
    task_type: TaskType
    timestamp: datetime
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "content_length": len(self.content),
            "provider": self.provider.value,
            "model_name": self.model_name,
            "task_type": self.task_type.value,
            "timestamp": self.timestamp.isoformat(),
            "usage": self.usage,
            "metadata": self.metadata,
        }


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    provider: ModelProvider
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    timeout: int = 60
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging (excluding sensitive data)."""
        return {
            "provider": self.provider.value,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }


@dataclass
class ModelRoutingRule:
    """Rule for routing tasks to specific models."""

    task_types: List[TaskType]
    primary_config: ModelConfig
    fallback_config: Optional[ModelConfig] = None

    def matches_task(self, task_type: TaskType) -> bool:
        """Check if this rule applies to the given task type."""
        return task_type in self.task_types
