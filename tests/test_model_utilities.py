"""
Tests for model interaction utilities.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from orchestrator.core.config import Config
from orchestrator.models.utilities import ModelInteractionManager
from orchestrator.models.types import TaskType, ModelProvider, ModelResponse
from orchestrator.models.templates import PromptTemplateManager
from orchestrator.models.parsers import ResponseParser, ParsedResponse, ParsedResponseType
from orchestrator.models.context import ContextManager, ContextStrategy


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
def mock_model_response():
    """Create a mock model response."""
    return ModelResponse(
        content="Test response content",
        provider=ModelProvider.OPENAI,
        model_name="gpt-4",
        task_type=TaskType.PLANNING,
        timestamp=datetime.now(),
        usage={"total_tokens": 100}
    )


@pytest.fixture
def mock_parsed_response():
    """Create a mock parsed response."""
    return ParsedResponse(
        response_type=ParsedResponseType.TEST_PLAN,
        content={"test_plan": {"objectives": ["Test objective 1"]}},
        validation_errors=None
    )


class TestModelInteractionManager:
    """Test cases for ModelInteractionManager."""
    
    @patch('orchestrator.models.utilities.ModelRouter')
    @patch('orchestrator.models.utilities.PromptTemplateManager')
    @patch('orchestrator.models.utilities.ResponseParser')
    @patch('orchestrator.models.utilities.ContextManager')
    @patch('orchestrator.models.utilities.ModelRateLimitManager')
    def test_initialization(self, mock_rate_manager, mock_context, mock_parser, 
                          mock_template, mock_router, mock_config):
        """Test manager initialization."""
        manager = ModelInteractionManager(mock_config)
        
        assert manager.config == mock_config
        mock_router.assert_called_once_with(mock_config)
        mock_template.assert_called_once()
        mock_parser.assert_called_once()
        mock_context.assert_called_once()
        mock_rate_manager.assert_called_once()
    
    @patch('orchestrator.models.utilities.ModelRouter')
    @patch('orchestrator.models.utilities.PromptTemplateManager')
    @patch('orchestrator.models.utilities.ResponseParser')
    @patch('orchestrator.models.utilities.ContextManager')
    @patch('orchestrator.models.utilities.ModelRateLimitManager')
    @pytest.mark.asyncio
    async def test_execute_task_success(self, mock_rate_manager, mock_context, 
                                      mock_parser, mock_template, mock_router, 
                                      mock_config, mock_model_response, mock_parsed_response):
        """Test successful task execution."""
        # Setup mocks
        mock_context_instance = Mock()
        mock_context_instance.prepare_context.return_value = {"prepared": "context"}
        mock_context_instance.validate_context_size.return_value = (True, 100, 1000)
        mock_context.return_value = mock_context_instance
        
        mock_template_instance = Mock()
        mock_template_instance.format_prompt.return_value = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User prompt"}
        ]
        mock_template.return_value = mock_template_instance
        
        mock_router_instance = Mock()
        mock_router_instance.route_task = AsyncMock(return_value=mock_model_response)
        mock_router.return_value = mock_router_instance
        
        mock_parser_instance = Mock()
        mock_parser_instance.parse_response.return_value = mock_parsed_response
        mock_parser.return_value = mock_parser_instance
        
        # Create manager and execute task
        manager = ModelInteractionManager(mock_config)
        context = {"test": "context"}
        
        result = await manager.execute_task(TaskType.PLANNING, context)
        
        # Verify result structure
        assert "parsed_response" in result
        assert "raw_response" in result
        assert "metadata" in result
        assert result["metadata"]["success"] is True
        assert result["metadata"]["task_type"] == "planning"
        
        # Verify method calls
        mock_context_instance.prepare_context.assert_called_once()
        mock_template_instance.format_prompt.assert_called_once()
        mock_router_instance.route_task.assert_called_once()
        mock_parser_instance.parse_response.assert_called_once()
    
    @patch('orchestrator.models.utilities.ModelRouter')
    @patch('orchestrator.models.utilities.PromptTemplateManager')
    @patch('orchestrator.models.utilities.ResponseParser')
    @patch('orchestrator.models.utilities.ContextManager')
    @patch('orchestrator.models.utilities.ModelRateLimitManager')
    @pytest.mark.asyncio
    async def test_execute_task_context_too_large(self, mock_rate_manager, mock_context, 
                                                 mock_parser, mock_template, mock_router, 
                                                 mock_config, mock_model_response, mock_parsed_response):
        """Test task execution with context that's too large."""
        # Setup mocks
        mock_context_instance = Mock()
        mock_context_instance.prepare_context.side_effect = [
            {"large": "context"},  # First call (prioritize strategy)
            {"small": "context"}   # Second call (truncate strategy)
        ]
        mock_context_instance.validate_context_size.side_effect = [
            (False, 2000, 1000),  # First validation fails
            (True, 500, 1000)     # Second validation passes
        ]
        mock_context.return_value = mock_context_instance
        
        mock_template_instance = Mock()
        mock_template_instance.format_prompt.return_value = [
            {"role": "user", "content": "Test prompt"}
        ]
        mock_template.return_value = mock_template_instance
        
        mock_router_instance = Mock()
        mock_router_instance.route_task = AsyncMock(return_value=mock_model_response)
        mock_router.return_value = mock_router_instance
        
        mock_parser_instance = Mock()
        mock_parser_instance.parse_response.return_value = mock_parsed_response
        mock_parser.return_value = mock_parser_instance
        
        # Create manager and execute task
        manager = ModelInteractionManager(mock_config)
        context = {"large": "context"}
        
        result = await manager.execute_task(TaskType.PLANNING, context)
        
        # Verify context was prepared twice (prioritize then truncate)
        assert mock_context_instance.prepare_context.call_count == 2
        calls = mock_context_instance.prepare_context.call_args_list
        assert calls[1][1]["strategy"] == ContextStrategy.TRUNCATE
        
        # Verify task still succeeded
        assert result["metadata"]["success"] is True
    
    @patch('orchestrator.models.utilities.ModelRouter')
    @patch('orchestrator.models.utilities.PromptTemplateManager')
    @patch('orchestrator.models.utilities.ResponseParser')
    @patch('orchestrator.models.utilities.ContextManager')
    @patch('orchestrator.models.utilities.ModelRateLimitManager')
    @pytest.mark.asyncio
    async def test_plan_tests(self, mock_rate_manager, mock_context, mock_parser, 
                            mock_template, mock_router, mock_config, mock_model_response, 
                            mock_parsed_response):
        """Test plan_tests convenience method."""
        # Setup mocks
        mock_context_instance = Mock()
        mock_context_instance.prepare_context.return_value = {"prepared": "context"}
        mock_context_instance.validate_context_size.return_value = (True, 100, 1000)
        mock_context.return_value = mock_context_instance
        
        mock_template_instance = Mock()
        mock_template_instance.format_prompt.return_value = [{"role": "user", "content": "prompt"}]
        mock_template.return_value = mock_template_instance
        
        mock_router_instance = Mock()
        mock_router_instance.route_task = AsyncMock(return_value=mock_model_response)
        mock_router.return_value = mock_router_instance
        
        mock_parser_instance = Mock()
        mock_parser_instance.parse_response.return_value = mock_parsed_response
        mock_parser.return_value = mock_parser_instance
        
        # Create manager and call plan_tests
        manager = ModelInteractionManager(mock_config)
        
        result = await manager.plan_tests(
            specification="Test spec",
            existing_tests="Existing tests",
            extra_param="extra"
        )
        
        # Verify the task was called with correct parameters
        mock_router_instance.route_task.assert_called_once()
        call_args = mock_router_instance.route_task.call_args
        assert call_args[1]["task_type"] == TaskType.PLANNING
        
        # Verify context was prepared with correct data
        prepare_call = mock_context_instance.prepare_context.call_args
        context_data = prepare_call[1]["context"]
        assert "specification" in context_data
        assert "existing_tests" in context_data
        assert "extra_param" in context_data
    
    @patch('orchestrator.models.utilities.ModelRouter')
    @patch('orchestrator.models.utilities.PromptTemplateManager')
    @patch('orchestrator.models.utilities.ResponseParser')
    @patch('orchestrator.models.utilities.ContextManager')
    @patch('orchestrator.models.utilities.ModelRateLimitManager')
    @pytest.mark.asyncio
    async def test_debug_failure(self, mock_rate_manager, mock_context, mock_parser, 
                               mock_template, mock_router, mock_config, mock_model_response, 
                               mock_parsed_response):
        """Test debug_failure convenience method."""
        # Setup mocks (similar to plan_tests)
        mock_context_instance = Mock()
        mock_context_instance.prepare_context.return_value = {"prepared": "context"}
        mock_context_instance.validate_context_size.return_value = (True, 100, 1000)
        mock_context.return_value = mock_context_instance
        
        mock_template_instance = Mock()
        mock_template_instance.format_prompt.return_value = [{"role": "user", "content": "prompt"}]
        mock_template.return_value = mock_template_instance
        
        mock_router_instance = Mock()
        mock_router_instance.route_task = AsyncMock(return_value=mock_model_response)
        mock_router.return_value = mock_router_instance
        
        mock_parser_instance = Mock()
        mock_parser_instance.parse_response.return_value = mock_parsed_response
        mock_parser.return_value = mock_parser_instance
        
        # Create manager and call debug_failure
        manager = ModelInteractionManager(mock_config)
        
        result = await manager.debug_failure(
            test_name="test_login",
            error_message="Element not found",
            stack_trace="Stack trace here",
            test_code="test code here",
            artifacts={"screenshots": True, "trace": False}
        )
        
        # Verify the task was called with DEBUGGING type
        call_args = mock_router_instance.route_task.call_args
        assert call_args[1]["task_type"] == TaskType.DEBUGGING
        
        # Verify context includes all debug-specific fields
        prepare_call = mock_context_instance.prepare_context.call_args
        context_data = prepare_call[1]["context"]
        assert "test_name" in context_data
        assert "error_message" in context_data
        assert "stack_trace" in context_data
        assert "test_code" in context_data
        assert "has_screenshots" in context_data
        assert "has_trace" in context_data
        assert context_data["has_screenshots"] is True
        assert context_data["has_trace"] is False
    
    @patch('orchestrator.models.utilities.ModelRouter')
    @patch('orchestrator.models.utilities.PromptTemplateManager')
    @patch('orchestrator.models.utilities.ResponseParser')
    @patch('orchestrator.models.utilities.ContextManager')
    @patch('orchestrator.models.utilities.ModelRateLimitManager')
    def test_get_system_status(self, mock_rate_manager, mock_context, mock_parser, 
                             mock_template, mock_router, mock_config):
        """Test get_system_status method."""
        # Setup mocks
        mock_router_instance = Mock()
        mock_router_instance.get_routing_info.return_value = {"routing": "info"}
        mock_router.return_value = mock_router_instance
        
        mock_template_instance = Mock()
        mock_template_instance.list_available_templates.return_value = [TaskType.PLANNING, TaskType.DEBUGGING]
        mock_template.return_value = mock_template_instance
        
        mock_rate_manager_instance = Mock()
        mock_rate_manager_instance.get_usage_stats.return_value = {"usage": "stats"}
        mock_rate_manager.return_value = mock_rate_manager_instance
        
        mock_context_instance = Mock()
        mock_context_instance.model_name = "gpt-4"
        mock_context.return_value = mock_context_instance
        
        # Create manager and get status
        manager = ModelInteractionManager(mock_config)
        status = manager.get_system_status()
        
        # Verify status structure
        assert "router" in status
        assert "templates" in status
        assert "rate_limits" in status
        assert "context_manager" in status
        
        assert status["router"] == {"routing": "info"}
        assert status["rate_limits"] == {"usage": "stats"}
        assert "available_templates" in status["templates"]
        assert "default_model" in status["context_manager"]
    
    @patch('orchestrator.models.utilities.ModelRouter')
    @patch('orchestrator.models.utilities.PromptTemplateManager')
    @patch('orchestrator.models.utilities.ResponseParser')
    @patch('orchestrator.models.utilities.ContextManager')
    @patch('orchestrator.models.utilities.ModelRateLimitManager')
    @pytest.mark.asyncio
    async def test_validate_setup(self, mock_rate_manager, mock_context, mock_parser, 
                                mock_template, mock_router, mock_config):
        """Test validate_setup method."""
        # Setup mocks
        mock_router_instance = Mock()
        mock_router_instance.get_available_providers.return_value = [ModelProvider.OPENAI]
        mock_router_instance.routing_rules = [Mock(), Mock()]
        mock_router.return_value = mock_router_instance
        
        mock_template_instance = Mock()
        mock_template_instance.list_available_templates.return_value = [TaskType.PLANNING]
        mock_template.return_value = mock_template_instance
        
        mock_context_instance = Mock()
        mock_context_instance.prepare_context.return_value = {"test": "prepared"}
        mock_context.return_value = mock_context_instance
        
        mock_rate_manager_instance = Mock()
        mock_rate_manager_instance.get_usage_stats.return_value = {"openai": {}}
        mock_rate_manager.return_value = mock_rate_manager_instance
        
        # Create manager and validate setup
        manager = ModelInteractionManager(mock_config)
        validation = await manager.validate_setup()
        
        # Verify validation structure
        assert "router" in validation
        assert "templates" in validation
        assert "context_manager" in validation
        assert "rate_limiter" in validation
        
        # All should be "ok" status
        assert validation["router"]["status"] == "ok"
        assert validation["templates"]["status"] == "ok"
        assert validation["context_manager"]["status"] == "ok"
        assert validation["rate_limiter"]["status"] == "ok"