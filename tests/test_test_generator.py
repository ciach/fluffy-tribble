"""
Unit tests for the test generator.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from orchestrator.generation.generator import TestGenerator
from orchestrator.generation.selector_auditor import SelectorAuditor
from orchestrator.generation.models import GeneratedTest, AuditResult, PageObjectScaffold
from orchestrator.planning.models import (
    TestPlan, TestCase, TestStep, Assertion, PageObject, SelectorInfo
)
from orchestrator.models.types import ModelResponse, TaskType, ModelProvider
from orchestrator.core.exceptions import ValidationError


@pytest.fixture
def mock_model_router():
    """Create a mock model router."""
    return Mock()


@pytest.fixture
def mock_selector_auditor():
    """Create a mock selector auditor."""
    auditor = Mock(spec=SelectorAuditor)
    # Default to compliant audit result
    auditor.audit_test_code.return_value = AuditResult(
        is_compliant=True,
        violations=[],
        total_selectors=3,
        compliant_selectors=3
    )
    auditor.validate_compliance.return_value = True
    return auditor


@pytest.fixture
def sample_test_plan():
    """Create a sample test plan."""
    test_step = TestStep(
        action="click",
        target="login button",
        description="Click the login button"
    )
    
    assertion = Assertion(
        type="url",
        target="current page",
        expected="/dashboard",
        description="User redirected to dashboard"
    )
    
    test_case = TestCase(
        name="test_successful_login",
        description="Test successful user login",
        steps=[test_step],
        assertions=[assertion],
        estimated_duration=30.0
    )
    
    selector_info = SelectorInfo(
        selector="getByRole('button', { name: 'Login' })",
        type="role",
        element_description="login button",
        is_compliant=True
    )
    
    page_object = PageObject(
        name="LoginPage",
        url_pattern="/login",
        selectors=[selector_info],
        methods=["login", "fillCredentials"]
    )
    
    return TestPlan(
        specification_id="test-001",
        test_cases=[test_case],
        page_objects=[page_object],
        estimated_duration=60.0
    )


@pytest.fixture
def sample_generated_test_content():
    """Sample generated test content."""
    return """
import { test, expect } from '@playwright/test';

test('test_successful_login', async ({ page }) => {
  await page.goto('/login');
  await page.getByRole('button', { name: 'Login' }).click();
  await expect(page).toHaveURL('/dashboard');
});
"""


class TestTestGenerator:
    """Test cases for TestGenerator."""
    
    def test_init(self, mock_model_router, mock_selector_auditor):
        """Test test generator initialization."""
        generator = TestGenerator(mock_model_router, mock_selector_auditor)
        
        assert generator.model_router == mock_model_router
        assert generator.selector_auditor == mock_selector_auditor
        assert generator._generation_prompts is not None
    
    def test_init_with_default_auditor(self, mock_model_router):
        """Test initialization with default selector auditor."""
        with patch('orchestrator.generation.generator.SelectorAuditor') as mock_auditor_class:
            generator = TestGenerator(mock_model_router)
            mock_auditor_class.assert_called_once()
    
    def test_generate_test_success(self, mock_model_router, mock_selector_auditor, 
                                 sample_test_plan, sample_generated_test_content):
        """Test successful test generation."""
        # Setup mock response
        mock_response = ModelResponse(
            content=sample_generated_test_content,
            provider=ModelProvider.OLLAMA,
            model_name="qwen3",
            task_type=TaskType.DRAFTING,
            timestamp=datetime.now()
        )
        mock_model_router.generate_response.return_value = mock_response
        
        generator = TestGenerator(mock_model_router, mock_selector_auditor)
        result = generator.generate_test(sample_test_plan, "login.spec.ts")
        
        assert isinstance(result, GeneratedTest)
        assert result.name == "login.spec.ts"
        assert result.content == sample_generated_test_content
        assert result.file_path == "e2e/login.spec.ts"
        assert result.test_plan_id == "test-001"
        assert "LoginPage" in result.page_objects
        
        # Verify model router was called with correct task type
        mock_model_router.generate_response.assert_called_once()
        call_args = mock_model_router.generate_response.call_args
        assert call_args[1]["task_type"] == TaskType.DRAFTING
        
        # Verify auditing was performed
        mock_selector_auditor.audit_test_code.assert_called_once_with(
            sample_generated_test_content, "e2e/login.spec.ts"
        )
        mock_selector_auditor.validate_compliance.assert_called_once()
    
    def test_generate_test_audit_failure(self, mock_model_router, mock_selector_auditor,
                                       sample_test_plan, sample_generated_test_content):
        """Test test generation with audit failure."""
        # Setup mock response
        mock_response = ModelResponse(
            content=sample_generated_test_content,
            provider=ModelProvider.OLLAMA,
            model_name="qwen3",
            task_type=TaskType.DRAFTING,
            timestamp=datetime.now()
        )
        mock_model_router.generate_response.return_value = mock_response
        
        # Setup audit failure
        mock_selector_auditor.validate_compliance.side_effect = ValidationError(
            "Selector policy violations found"
        )
        
        generator = TestGenerator(mock_model_router, mock_selector_auditor)
        
        with pytest.raises(ValidationError, match="Selector policy violations found"):
            generator.generate_test(sample_test_plan, "login.spec.ts")
    
    def test_update_test_success(self, mock_model_router, mock_selector_auditor):
        """Test successful test update."""
        existing_test = """
import { test, expect } from '@playwright/test';

test('old test', async ({ page }) => {
  await page.goto('/login');
});
"""
        
        updated_test = """
import { test, expect } from '@playwright/test';

test('updated test', async ({ page }) => {
  await page.goto('/login');
  await page.getByRole('button', { name: 'Login' }).click();
});
"""
        
        mock_response = ModelResponse(
            content=updated_test,
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            task_type=TaskType.GENERATION,
            timestamp=datetime.now()
        )
        mock_model_router.generate_response.return_value = mock_response
        
        generator = TestGenerator(mock_model_router, mock_selector_auditor)
        changes = ["Add login button click"]
        
        result = generator.update_test(existing_test, changes, "e2e/login.spec.ts")
        
        assert isinstance(result, GeneratedTest)
        assert result.content == updated_test
        assert result.file_path == "e2e/login.spec.ts"
        assert result.test_plan_id == "updated"
        
        # Verify correct task type was used
        call_args = mock_model_router.generate_response.call_args
        assert call_args[1]["task_type"] == TaskType.GENERATION
    
    def test_scaffold_page_object(self, mock_model_router, mock_selector_auditor, sample_test_plan):
        """Test page object scaffolding."""
        page_object_content = """
import { Page, Locator } from '@playwright/test';

export class LoginPage {
  constructor(private page: Page) {}
  
  get loginButton(): Locator {
    return this.page.getByRole('button', { name: 'Login' });
  }
  
  async login(username: string, password: string): Promise<void> {
    // Implementation
  }
}
"""
        
        mock_response = ModelResponse(
            content=page_object_content,
            provider=ModelProvider.OLLAMA,
            model_name="qwen3",
            task_type=TaskType.DRAFTING,
            timestamp=datetime.now()
        )
        mock_model_router.generate_response.return_value = mock_response
        
        generator = TestGenerator(mock_model_router, mock_selector_auditor)
        page_object = sample_test_plan.page_objects[0]
        
        result = generator.scaffold_page_object(page_object)
        
        assert isinstance(result, PageObjectScaffold)
        assert result.class_name == "LoginPage"
        assert result.file_path == "e2e/pages/login.page.ts"
        assert result.url_pattern == "/login"
        assert "login button" in result.selectors
        assert "login" in result.methods
        assert len(result.imports) > 0
    
    def test_format_test_plan(self, mock_model_router, mock_selector_auditor, sample_test_plan):
        """Test test plan formatting for prompts."""
        generator = TestGenerator(mock_model_router, mock_selector_auditor)
        formatted = generator._format_test_plan(sample_test_plan)
        
        assert "test-001" in formatted
        assert "test_successful_login" in formatted
        assert "click: login button" in formatted
        assert "LoginPage" in formatted
        assert "60.0s" in formatted
    
    def test_format_page_object(self, mock_model_router, mock_selector_auditor, sample_test_plan):
        """Test page object formatting for prompts."""
        generator = TestGenerator(mock_model_router, mock_selector_auditor)
        page_object = sample_test_plan.page_objects[0]
        formatted = generator._format_page_object(page_object)
        
        assert "LoginPage" in formatted
        assert "/login" in formatted
        assert "login button" in formatted
        assert "getByRole" in formatted
        assert "login, fillCredentials" in formatted
    
    def test_extract_page_objects(self, mock_model_router, mock_selector_auditor, sample_test_plan):
        """Test page object extraction from test plan."""
        generator = TestGenerator(mock_model_router, mock_selector_auditor)
        page_objects = generator._extract_page_objects(sample_test_plan)
        
        assert page_objects == ["LoginPage"]
    
    def test_extract_imports(self, mock_model_router, mock_selector_auditor):
        """Test import extraction from test content."""
        test_content = """
import { test, expect } from '@playwright/test';
import { LoginPage } from './pages/login.page';

test('login test', async ({ page }) => {
  // test content
});
"""
        
        generator = TestGenerator(mock_model_router, mock_selector_auditor)
        imports = generator._extract_imports(test_content)
        
        assert len(imports) == 2
        assert "import { test, expect } from '@playwright/test';" in imports
        assert "import { LoginPage } from './pages/login.page';" in imports
    
    def test_model_router_error_handling(self, mock_model_router, mock_selector_auditor, sample_test_plan):
        """Test error handling when model router fails."""
        mock_model_router.generate_response.side_effect = Exception("Model unavailable")
        
        generator = TestGenerator(mock_model_router, mock_selector_auditor)
        
        with pytest.raises(ValidationError, match="Test generation failed"):
            generator.generate_test(sample_test_plan, "test.spec.ts")
    
    def test_custom_output_directory(self, mock_model_router, mock_selector_auditor, 
                                   sample_test_plan, sample_generated_test_content):
        """Test test generation with custom output directory."""
        mock_response = ModelResponse(
            content=sample_generated_test_content,
            provider=ModelProvider.OLLAMA,
            model_name="qwen3",
            task_type=TaskType.DRAFTING,
            timestamp=datetime.now()
        )
        mock_model_router.generate_response.return_value = mock_response
        
        generator = TestGenerator(mock_model_router, mock_selector_auditor)
        result = generator.generate_test(sample_test_plan, "login.spec.ts", "tests/e2e")
        
        assert result.file_path == "tests/e2e/login.spec.ts"