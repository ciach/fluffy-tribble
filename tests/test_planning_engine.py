"""
Unit tests for the planning engine.
"""

import json
import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from orchestrator.planning.engine import PlanningEngine
from orchestrator.planning.models import (
    TestSpecification, TestPlan, TestCase, TestGapAnalysis, Priority
)
from orchestrator.models.types import ModelResponse, TaskType, ModelProvider
from orchestrator.core.exceptions import PlanningError


@pytest.fixture
def mock_model_router():
    """Create a mock model router."""
    router = Mock()
    return router


@pytest.fixture
def sample_specification():
    """Create a sample test specification."""
    return TestSpecification(
        id="test-001",
        name="User Login Flow",
        description="Test user authentication and login process",
        requirements=[
            "User can enter username and password",
            "User can submit login form",
            "User sees success message on valid login",
            "User sees error message on invalid login"
        ],
        priority=Priority.HIGH,
        tags=["authentication", "login"]
    )


@pytest.fixture
def sample_analysis_response():
    """Sample analysis response from model."""
    return {
        "core_functionality": ["login", "authentication", "form_validation"],
        "user_workflows": ["successful_login", "failed_login", "password_reset"],
        "edge_cases": ["empty_fields", "invalid_credentials", "network_error"],
        "setup_requirements": ["test_user_account", "clean_database"],
        "priority_assessment": "high",
        "estimated_complexity": "moderate"
    }


@pytest.fixture
def sample_test_plan_response():
    """Sample test plan response from model."""
    return {
        "test_cases": [
            {
                "name": "test_successful_login",
                "description": "Verify user can login with valid credentials",
                "steps": [
                    {
                        "action": "navigate",
                        "target": "login page",
                        "description": "Navigate to login page"
                    },
                    {
                        "action": "fill",
                        "target": "username field",
                        "value": "testuser",
                        "description": "Enter username"
                    },
                    {
                        "action": "fill",
                        "target": "password field", 
                        "value": "password123",
                        "description": "Enter password"
                    },
                    {
                        "action": "click",
                        "target": "login button",
                        "description": "Click login button"
                    }
                ],
                "assertions": [
                    {
                        "type": "url",
                        "target": "current page",
                        "expected": "/dashboard",
                        "description": "User redirected to dashboard"
                    }
                ],
                "setup_requirements": ["test_user_account"],
                "estimated_duration": 30.0
            }
        ],
        "page_objects": [
            {
                "name": "LoginPage",
                "url_pattern": "/login",
                "selectors": [
                    {
                        "selector": "getByRole('textbox', { name: 'Username' })",
                        "type": "role",
                        "element_description": "username input field",
                        "is_compliant": True
                    }
                ],
                "methods": ["login", "fillCredentials"]
            }
        ],
        "setup_requirements": ["test_database"],
        "estimated_duration": 120.0
    }


class TestPlanningEngine:
    """Test cases for PlanningEngine."""
    
    def test_init(self, mock_model_router):
        """Test planning engine initialization."""
        engine = PlanningEngine(mock_model_router)
        assert engine.model_router == mock_model_router
        assert engine._planning_prompts is not None
        assert "analyze_specification" in engine._planning_prompts
    
    def test_analyze_specification_success(self, mock_model_router, sample_specification, 
                                         sample_analysis_response):
        """Test successful specification analysis."""
        # Setup mock response
        mock_response = ModelResponse(
            content=json.dumps(sample_analysis_response),
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            task_type=TaskType.PLANNING,
            timestamp=datetime.now()
        )
        mock_model_router.generate_response.return_value = mock_response
        
        engine = PlanningEngine(mock_model_router)
        result = engine.analyze_specification(sample_specification)
        
        assert result == sample_analysis_response
        mock_model_router.generate_response.assert_called_once()
        call_args = mock_model_router.generate_response.call_args
        assert call_args[1]["task_type"] == TaskType.PLANNING
    
    def test_analyze_specification_json_error(self, mock_model_router, sample_specification):
        """Test specification analysis with invalid JSON response."""
        # Setup mock response with invalid JSON
        mock_response = ModelResponse(
            content="invalid json response",
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            task_type=TaskType.PLANNING,
            timestamp=datetime.now()
        )
        mock_model_router.generate_response.return_value = mock_response
        
        engine = PlanningEngine(mock_model_router)
        
        with pytest.raises(PlanningError, match="Invalid analysis response format"):
            engine.analyze_specification(sample_specification)
    
    def test_create_test_plan_success(self, mock_model_router, sample_specification,
                                    sample_analysis_response, sample_test_plan_response):
        """Test successful test plan creation."""
        # Setup mock responses for both analysis and plan generation
        analysis_response = ModelResponse(
            content=json.dumps(sample_analysis_response),
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            task_type=TaskType.PLANNING,
            timestamp=datetime.now()
        )
        
        plan_response = ModelResponse(
            content=json.dumps(sample_test_plan_response),
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            task_type=TaskType.PLANNING,
            timestamp=datetime.now()
        )
        
        mock_model_router.generate_response.side_effect = [analysis_response, plan_response]
        
        engine = PlanningEngine(mock_model_router)
        result = engine.create_test_plan(sample_specification)
        
        assert isinstance(result, TestPlan)
        assert result.specification_id == sample_specification.id
        assert len(result.test_cases) == 1
        assert len(result.page_objects) == 1
        assert result.test_cases[0].name == "test_successful_login"
        assert len(result.test_cases[0].steps) == 4
        assert len(result.test_cases[0].assertions) == 1
    
    def test_identify_test_gaps_success(self, mock_model_router, sample_specification):
        """Test successful test gap analysis."""
        gap_response_data = {
            "missing_test_cases": ["test_password_reset", "test_account_lockout"],
            "incomplete_coverage": ["error_handling", "edge_cases"],
            "suggested_additions": ["test_remember_me_functionality"],
            "priority_gaps": ["test_password_reset"]
        }
        
        mock_response = ModelResponse(
            content=json.dumps(gap_response_data),
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            task_type=TaskType.ANALYSIS,
            timestamp=datetime.now()
        )
        mock_model_router.generate_response.return_value = mock_response
        
        existing_tests = ["test('login with valid credentials', async () => { ... })"]
        
        engine = PlanningEngine(mock_model_router)
        result = engine.identify_test_gaps(sample_specification, existing_tests)
        
        assert isinstance(result, TestGapAnalysis)
        assert len(result.missing_test_cases) == 2
        assert "test_password_reset" in result.missing_test_cases
        assert len(result.priority_gaps) == 1
    
    def test_convert_to_test_plan(self, mock_model_router, sample_test_plan_response):
        """Test conversion of JSON data to TestPlan object."""
        engine = PlanningEngine(mock_model_router)
        result = engine._convert_to_test_plan("test-001", sample_test_plan_response)
        
        assert isinstance(result, TestPlan)
        assert result.specification_id == "test-001"
        assert len(result.test_cases) == 1
        
        test_case = result.test_cases[0]
        assert test_case.name == "test_successful_login"
        assert len(test_case.steps) == 4
        assert test_case.steps[0].action == "navigate"
        assert test_case.steps[1].value == "testuser"
        
        assert len(test_case.assertions) == 1
        assert test_case.assertions[0].type == "url"
        assert test_case.assertions[0].expected == "/dashboard"
        
        assert len(result.page_objects) == 1
        page_obj = result.page_objects[0]
        assert page_obj.name == "LoginPage"
        assert len(page_obj.selectors) == 1
        assert page_obj.selectors[0].is_compliant is True
    
    def test_model_router_error_handling(self, mock_model_router, sample_specification):
        """Test error handling when model router fails."""
        mock_model_router.generate_response.side_effect = Exception("Model unavailable")
        
        engine = PlanningEngine(mock_model_router)
        
        with pytest.raises(PlanningError, match="Analysis failed"):
            engine.analyze_specification(sample_specification)
    
    def test_empty_test_plan_handling(self, mock_model_router, sample_specification):
        """Test handling of empty test plan response."""
        empty_plan = {
            "test_cases": [],
            "page_objects": [],
            "setup_requirements": [],
            "estimated_duration": 0.0
        }
        
        analysis_response = ModelResponse(
            content=json.dumps({"core_functionality": []}),
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            task_type=TaskType.PLANNING,
            timestamp=datetime.now()
        )
        
        plan_response = ModelResponse(
            content=json.dumps(empty_plan),
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            task_type=TaskType.PLANNING,
            timestamp=datetime.now()
        )
        
        mock_model_router.generate_response.side_effect = [analysis_response, plan_response]
        
        engine = PlanningEngine(mock_model_router)
        result = engine.create_test_plan(sample_specification)
        
        assert isinstance(result, TestPlan)
        assert len(result.test_cases) == 0
        assert len(result.page_objects) == 0