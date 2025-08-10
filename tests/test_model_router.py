"""
Tests for the model router implementation.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from orchestrator.core.config import Config
from orchestrator.models.router import ModelRouter
from orchestrator.models.types import TaskType, ModelProvider, ModelResponse
from orchestrator.core.exceptions import ModelError, ValidationError


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = Mock(spec=Config)
    config.model_provider = "mixed"
    config.openai_api_key = "test-key"
    config.ollama_base_url = "http://localhost:11434"
    config.debug_enabled = False
    return config


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    client = Mock()
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = "Test response"
    response.usage = Mock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 20
    response.usage.total_tokens = 30
    response.model = "gpt-4"
    
    client.chat.completions.create.return_value = response
    return client


@pytest.fixture
def mock_litellm_response():
    """Create a mock LiteLLM response."""
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = "Ollama test response"
    response.usage = Mock()
    response.usage.prompt_tokens = 15
    response.usage.completion_tokens = 25
    response.usage.total_tokens = 40
    response.model = "qwen2.5:7b"
    return response


class TestModelRouter:
    """Test cases for ModelRouter."""
    
    @patch('orchestrator.models.router.OpenAI')
    def test_initialization(self, mock_openai_class, mock_config):
        """Test router initialization."""
        mock_openai_class.return_value = Mock()
        
        router = ModelRouter(mock_config)
        
        assert router.config == mock_config
        assert router.openai_client is not None
        assert len(router.routing_rules) > 0
        mock_openai_class.assert_called_once_with(api_key="test-key")
    
    @patch('orchestrator.models.router.OpenAI')
    def test_initialization_without_openai_key(self, mock_openai_class, mock_config):
        """Test router initialization without OpenAI key."""
        mock_config.openai_api_key = None
        
        router = ModelRouter(mock_config)
        
        assert router.openai_client is None
        mock_openai_class.assert_not_called()
    
    @patch('orchestrator.models.router.OpenAI')
    def test_find_routing_rule(self, mock_openai_class, mock_config):
        """Test finding routing rules for different task types."""
        mock_openai_class.return_value = Mock()
        router = ModelRouter(mock_config)
        
        planning_rule = router._find_routing_rule(TaskType.PLANNING)
        drafting_rule = router._find_routing_rule(TaskType.DRAFTING)
        
        assert planning_rule is not None
        assert drafting_rule is not None
        assert TaskType.PLANNING in planning_rule.task_types
        assert TaskType.DRAFTING in drafting_rule.task_types
    
    @patch('orchestrator.models.router.OpenAI')
    @pytest.mark.asyncio
    async def test_call_openai_model(self, mock_openai_class, mock_config, mock_openai_client):
        """Test calling OpenAI model."""
        mock_openai_class.return_value = mock_openai_client
        router = ModelRouter(mock_config)
        
        messages = [{"role": "user", "content": "Test message"}]
        config = router.routing_rules[0].primary_config
        
        result = await router._call_openai_model(config, messages)
        
        assert result["content"] == "Test response"
        assert result["usage"]["total_tokens"] == 30
        assert result["model"] == "gpt-4"
        mock_openai_client.chat.completions.create.assert_called_once()
    
    @patch('orchestrator.models.router.OpenAI')
    @patch('orchestrator.models.router.litellm.acompletion')
    @pytest.mark.asyncio
    async def test_call_ollama_model(self, mock_litellm, mock_openai_class, mock_config, mock_litellm_response):
        """Test calling Ollama model via LiteLLM."""
        mock_openai_class.return_value = Mock()
        mock_litellm.return_value = mock_litellm_response
        
        router = ModelRouter(mock_config)
        
        messages = [{"role": "user", "content": "Test message"}]
        config = router.routing_rules[1].primary_config  # Ollama config for drafting
        
        result = await router._call_ollama_model(config, messages)
        
        assert result["content"] == "Ollama test response"
        assert result["usage"]["total_tokens"] == 40
        mock_litellm.assert_called_once()
    
    @patch('orchestrator.models.router.OpenAI')
    @pytest.mark.asyncio
    async def test_route_task_planning(self, mock_openai_class, mock_config, mock_openai_client):
        """Test routing a planning task to OpenAI."""
        mock_openai_class.return_value = mock_openai_client
        router = ModelRouter(mock_config)
        
        messages = [{"role": "user", "content": "Create a test plan"}]
        
        response = await router.route_task(TaskType.PLANNING, messages)
        
        assert isinstance(response, ModelResponse)
        assert response.content == "Test response"
        assert response.provider == ModelProvider.OPENAI
        assert response.task_type == TaskType.PLANNING
        assert isinstance(response.timestamp, datetime)
    
    @patch('orchestrator.models.router.OpenAI')
    @patch('orchestrator.models.router.litellm.acompletion')
    @pytest.mark.asyncio
    async def test_route_task_drafting(self, mock_litellm, mock_openai_class, mock_config, mock_litellm_response):
        """Test routing a drafting task to Ollama."""
        mock_openai_class.return_value = Mock()
        mock_litellm.return_value = mock_litellm_response
        
        router = ModelRouter(mock_config)
        
        messages = [{"role": "user", "content": "Draft a test"}]
        
        response = await router.route_task(TaskType.DRAFTING, messages)
        
        assert isinstance(response, ModelResponse)
        assert response.content == "Ollama test response"
        assert response.provider == ModelProvider.OLLAMA
        assert response.task_type == TaskType.DRAFTING
    
    @patch('orchestrator.models.router.OpenAI')
    @pytest.mark.asyncio
    async def test_route_task_empty_messages(self, mock_openai_class, mock_config):
        """Test routing with empty messages raises ValidationError."""
        mock_openai_class.return_value = Mock()
        router = ModelRouter(mock_config)
        
        with pytest.raises(ValidationError):
            await router.route_task(TaskType.PLANNING, [])
    
    @patch('orchestrator.models.router.OpenAI')
    @pytest.mark.asyncio
    async def test_route_task_with_fallback(self, mock_openai_class, mock_config, mock_openai_client):
        """Test fallback behavior when primary model fails."""
        mock_openai_class.return_value = mock_openai_client
        
        # Make primary model fail, fallback succeed
        mock_openai_client.chat.completions.create.side_effect = [
            Exception("Primary failed"),
            Mock(choices=[Mock(message=Mock(content="Fallback response"))], 
                 usage=Mock(prompt_tokens=5, completion_tokens=10, total_tokens=15),
                 model="gpt-4")
        ]
        
        router = ModelRouter(mock_config)
        
        messages = [{"role": "user", "content": "Test message"}]
        
        response = await router.route_task(TaskType.DRAFTING, messages)
        
        assert response.content == "Fallback response"
        assert response.metadata["attempt"] == "fallback"
        assert "primary_error" in response.metadata
    
    @patch('orchestrator.models.router.OpenAI')
    def test_get_available_providers(self, mock_openai_class, mock_config):
        """Test getting available providers."""
        mock_openai_class.return_value = Mock()
        router = ModelRouter(mock_config)
        
        providers = router.get_available_providers()
        
        assert ModelProvider.OPENAI in providers
        assert ModelProvider.OLLAMA in providers
    
    @patch('orchestrator.models.router.OpenAI')
    def test_get_routing_info(self, mock_openai_class, mock_config):
        """Test getting routing information."""
        mock_openai_class.return_value = Mock()
        router = ModelRouter(mock_config)
        
        info = router.get_routing_info()
        
        assert info["provider_mode"] == "mixed"
        assert "available_providers" in info
        assert "routing_rules" in info
        assert len(info["routing_rules"]) > 0