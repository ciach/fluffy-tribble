"""
Model router implementation with LiteLLM integration.

Handles routing tasks to appropriate AI models based on task type,
with fallback handling and retry logic.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

import litellm
from openai import OpenAI

from ..core.config import Config
from ..core.exceptions import ModelError, ValidationError
from .types import (
    TaskType, 
    ModelProvider, 
    ModelResponse, 
    ModelConfig, 
    ModelRoutingRule
)

logger = logging.getLogger(__name__)


class ModelRouter:
    """
    Routes AI model requests to appropriate providers based on task type.
    
    Implements the dual-model architecture:
    - OpenAI for planning and debugging (high-quality reasoning)
    - Local Ollama for drafting (cost-effective generation)
    """
    
    def __init__(self, config: Config):
        """Initialize the model router with configuration."""
        self.config = config
        self.openai_client: Optional[OpenAI] = None
        self.routing_rules: List[ModelRoutingRule] = []
        self._setup_clients()
        self._setup_routing_rules()
    
    def _setup_clients(self) -> None:
        """Set up API clients for different providers."""
        try:
            # Set up OpenAI client
            if self.config.openai_api_key:
                self.openai_client = OpenAI(api_key=self.config.openai_api_key)
                logger.info("OpenAI client initialized successfully")
            else:
                logger.warning("OpenAI API key not provided, OpenAI models unavailable")
            
            # Configure LiteLLM for Ollama
            if self.config.model_provider in ["ollama", "mixed"]:
                litellm.set_verbose = self.config.debug_enabled
                logger.info(f"LiteLLM configured for Ollama at {self.config.ollama_base_url}")
                
        except Exception as e:
            logger.error(f"Failed to setup model clients: {e}")
            raise ModelError(f"Model client setup failed: {e}")
    
    def _setup_routing_rules(self) -> None:
        """Set up routing rules based on configuration."""
        try:
            # OpenAI configuration for high-quality reasoning tasks
            openai_config = ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4",
                api_key=self.config.openai_api_key,
                max_tokens=4000,
                temperature=0.3,  # Lower temperature for more consistent reasoning
                timeout=60,
                max_retries=3
            )
            
            # Ollama configuration for cost-effective drafting
            ollama_config = ModelConfig(
                provider=ModelProvider.OLLAMA,
                model_name="qwen2.5:7b",  # Qwen3 equivalent
                base_url=self.config.ollama_base_url,
                max_tokens=2000,
                temperature=0.7,  # Higher temperature for creative generation
                timeout=45,
                max_retries=2
            )
            
            # Set up routing rules based on provider configuration
            if self.config.model_provider == "openai":
                # Use OpenAI for all tasks
                self.routing_rules = [
                    ModelRoutingRule(
                        task_types=[TaskType.PLANNING, TaskType.DEBUGGING, TaskType.ANALYSIS],
                        primary_config=openai_config
                    ),
                    ModelRoutingRule(
                        task_types=[TaskType.DRAFTING, TaskType.GENERATION],
                        primary_config=openai_config
                    )
                ]
                
            elif self.config.model_provider == "ollama":
                # Use Ollama for all tasks (with OpenAI fallback if available)
                fallback = openai_config if self.config.openai_api_key else None
                self.routing_rules = [
                    ModelRoutingRule(
                        task_types=[TaskType.PLANNING, TaskType.DEBUGGING, TaskType.ANALYSIS],
                        primary_config=ollama_config,
                        fallback_config=fallback
                    ),
                    ModelRoutingRule(
                        task_types=[TaskType.DRAFTING, TaskType.GENERATION],
                        primary_config=ollama_config,
                        fallback_config=fallback
                    )
                ]
                
            else:  # mixed mode (default)
                # Use OpenAI for reasoning, Ollama for drafting
                self.routing_rules = [
                    ModelRoutingRule(
                        task_types=[TaskType.PLANNING, TaskType.DEBUGGING, TaskType.ANALYSIS],
                        primary_config=openai_config,
                        fallback_config=ollama_config
                    ),
                    ModelRoutingRule(
                        task_types=[TaskType.DRAFTING, TaskType.GENERATION],
                        primary_config=ollama_config,
                        fallback_config=openai_config
                    )
                ]
            
            logger.info(f"Model routing configured with {len(self.routing_rules)} rules")
            
        except Exception as e:
            logger.error(f"Failed to setup routing rules: {e}")
            raise ModelError(f"Routing rules setup failed: {e}")
    
    def _find_routing_rule(self, task_type: TaskType) -> Optional[ModelRoutingRule]:
        """Find the routing rule for a given task type."""
        for rule in self.routing_rules:
            if rule.matches_task(task_type):
                return rule
        return None
    
    async def _call_openai_model(
        self, 
        config: ModelConfig, 
        messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Call OpenAI model with the given configuration."""
        if not self.openai_client:
            raise ModelError("OpenAI client not initialized")
        
        try:
            response = self.openai_client.chat.completions.create(
                model=config.model_name,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                timeout=config.timeout
            )
            
            return {
                "content": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if response.usage else None,
                "model": response.model
            }
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise ModelError(f"OpenAI model call failed: {e}")
    
    async def _call_ollama_model(
        self, 
        config: ModelConfig, 
        messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Call Ollama model via LiteLLM."""
        try:
            # Format model name for LiteLLM
            model_name = f"ollama/{config.model_name}"
            
            response = await litellm.acompletion(
                model=model_name,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                api_base=config.base_url,
                timeout=config.timeout
            )
            
            return {
                "content": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                },
                "model": response.model
            }
            
        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            raise ModelError(f"Ollama model call failed: {e}")
    
    async def _call_model_with_config(
        self, 
        config: ModelConfig, 
        messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Call a model with the given configuration."""
        if config.provider == ModelProvider.OPENAI:
            return await self._call_openai_model(config, messages)
        elif config.provider == ModelProvider.OLLAMA:
            return await self._call_ollama_model(config, messages)
        else:
            raise ModelError(f"Unsupported model provider: {config.provider}")
    
    async def _call_model_with_retry(
        self, 
        config: ModelConfig, 
        messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Call model with retry logic."""
        last_error = None
        
        for attempt in range(config.max_retries):
            try:
                return await self._call_model_with_config(config, messages)
                
            except Exception as e:
                last_error = e
                if attempt < config.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(
                        f"Model call attempt {attempt + 1} failed, retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All {config.max_retries} model call attempts failed")
        
        raise ModelError(f"Model call failed after {config.max_retries} attempts: {last_error}")
    
    async def route_task(
        self, 
        task_type: TaskType, 
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None
    ) -> ModelResponse:
        """
        Route a task to the appropriate model based on task type.
        
        Args:
            task_type: The type of task to route
            messages: List of messages in OpenAI format
            context: Optional context information for routing decisions
            
        Returns:
            ModelResponse with the model's output
            
        Raises:
            ModelError: If all model calls fail
            ValidationError: If input validation fails
        """
        if not messages:
            raise ValidationError("Messages cannot be empty", validation_type="input")
        
        # Find routing rule
        rule = self._find_routing_rule(task_type)
        if not rule:
            raise ModelError(f"No routing rule found for task type: {task_type}")
        
        logger.info(f"Routing {task_type.value} task to {rule.primary_config.provider.value}")
        
        # Try primary model
        try:
            result = await self._call_model_with_retry(rule.primary_config, messages)
            
            response = ModelResponse(
                content=result["content"],
                provider=rule.primary_config.provider,
                model_name=result.get("model", rule.primary_config.model_name),
                task_type=task_type,
                timestamp=datetime.now(),
                usage=result.get("usage"),
                metadata={"attempt": "primary", "context": context}
            )
            
            logger.info(f"Successfully completed {task_type.value} task with {rule.primary_config.provider.value}")
            return response
            
        except ModelError as e:
            logger.warning(f"Primary model failed for {task_type.value}: {e}")
            
            # Try fallback model if available
            if rule.fallback_config:
                logger.info(f"Attempting fallback to {rule.fallback_config.provider.value}")
                
                try:
                    result = await self._call_model_with_retry(rule.fallback_config, messages)
                    
                    response = ModelResponse(
                        content=result["content"],
                        provider=rule.fallback_config.provider,
                        model_name=result.get("model", rule.fallback_config.model_name),
                        task_type=task_type,
                        timestamp=datetime.now(),
                        usage=result.get("usage"),
                        metadata={"attempt": "fallback", "primary_error": str(e), "context": context}
                    )
                    
                    logger.info(f"Successfully completed {task_type.value} task with fallback {rule.fallback_config.provider.value}")
                    return response
                    
                except ModelError as fallback_error:
                    logger.error(f"Fallback model also failed: {fallback_error}")
                    raise ModelError(f"Both primary and fallback models failed. Primary: {e}, Fallback: {fallback_error}")
            else:
                raise e
    
    def get_available_providers(self) -> List[ModelProvider]:
        """Get list of currently available model providers."""
        providers = []
        
        if self.openai_client:
            providers.append(ModelProvider.OPENAI)
        
        # Check Ollama availability (simplified check)
        if self.config.model_provider in ["ollama", "mixed"]:
            providers.append(ModelProvider.OLLAMA)
        
        return providers
    
    def get_routing_info(self) -> Dict[str, Any]:
        """Get information about current routing configuration."""
        return {
            "provider_mode": self.config.model_provider,
            "available_providers": [p.value for p in self.get_available_providers()],
            "routing_rules": [
                {
                    "task_types": [t.value for t in rule.task_types],
                    "primary_provider": rule.primary_config.provider.value,
                    "primary_model": rule.primary_config.model_name,
                    "has_fallback": rule.fallback_config is not None,
                    "fallback_provider": rule.fallback_config.provider.value if rule.fallback_config else None
                }
                for rule in self.routing_rules
            ]
        }