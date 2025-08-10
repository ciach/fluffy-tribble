"""
Prompt templates for different task types in the QA Operator.

Provides structured prompts for planning, debugging, drafting, and analysis tasks.
"""

from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass

from .types import TaskType


class PromptRole(Enum):
    """Roles for prompt messages."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class PromptTemplate:
    """Template for generating prompts for specific task types."""
    
    task_type: TaskType
    system_prompt: str
    user_template: str
    context_keys: List[str]
    max_context_length: int = 8000
    
    def format(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Format the template with provided context.
        
        Args:
            context: Dictionary containing values for template variables
            
        Returns:
            List of messages in OpenAI format
            
        Raises:
            KeyError: If required context keys are missing
        """
        # Validate required context keys
        missing_keys = [key for key in self.context_keys if key not in context]
        if missing_keys:
            raise KeyError(f"Missing required context keys: {missing_keys}")
        
        # Format user message with context
        user_message = self.user_template.format(**context)
        
        return [
            {"role": PromptRole.SYSTEM.value, "content": self.system_prompt},
            {"role": PromptRole.USER.value, "content": user_message}
        ]


class PromptTemplateManager:
    """Manages prompt templates for different task types."""
    
    def __init__(self):
        """Initialize with default templates."""
        self.templates: Dict[TaskType, PromptTemplate] = {}
        self._setup_default_templates()
    
    def _setup_default_templates(self) -> None:
        """Set up default prompt templates for all task types."""
        
        # Planning template
        self.templates[TaskType.PLANNING] = PromptTemplate(
            task_type=TaskType.PLANNING,
            system_prompt="""You are an expert QA engineer specializing in test planning and strategy.
Your role is to analyze test specifications and create comprehensive, actionable test plans.

Key responsibilities:
- Analyze requirements and identify test scenarios
- Create structured test plans with clear objectives
- Identify potential edge cases and failure modes
- Suggest appropriate testing strategies and approaches
- Consider both functional and non-functional requirements

Always provide detailed, actionable plans that can be implemented by test automation engineers.""",
            user_template="""Please analyze the following test specification and create a comprehensive test plan:

**Test Specification:**
{specification}

**Existing Test Coverage:**
{existing_tests}

**Requirements:**
- Create a structured test plan with clear test cases
- Identify gaps in current test coverage
- Suggest test data requirements and setup needs
- Consider edge cases and error scenarios
- Provide estimated effort and priority levels

Please format your response as a structured test plan with sections for:
1. Test Objectives
2. Test Cases (with steps and expected results)
3. Test Data Requirements
4. Coverage Analysis
5. Risk Assessment""",
            context_keys=["specification", "existing_tests"],
            max_context_length=6000
        )
        
        # Debugging template
        self.templates[TaskType.DEBUGGING] = PromptTemplate(
            task_type=TaskType.DEBUGGING,
            system_prompt="""You are an expert test automation engineer specializing in debugging Playwright test failures.
Your role is to analyze test failures and provide precise, actionable solutions.

Key responsibilities:
- Analyze stack traces, error messages, and test artifacts
- Identify root causes of test failures
- Suggest minimal, targeted fixes
- Distinguish between test issues and application issues
- Provide specific code changes when possible

Focus on practical solutions that address the immediate failure while improving test stability.""",
            user_template="""Please analyze this test failure and provide a solution:

**Test Name:** {test_name}

**Error Message:**
{error_message}

**Stack Trace:**
{stack_trace}

**Test Code:**
{test_code}

**Available Artifacts:**
- Screenshots: {has_screenshots}
- Trace: {has_trace}
- Console Logs: {has_console_logs}
- Network Logs: {has_network_logs}

**Additional Context:**
{additional_context}

Please provide:
1. Root cause analysis
2. Specific code changes needed
3. Explanation of why the failure occurred
4. Suggestions to prevent similar failures
5. Confidence level in the proposed solution (1-10)""",
            context_keys=["test_name", "error_message", "stack_trace", "test_code", 
                         "has_screenshots", "has_trace", "has_console_logs", 
                         "has_network_logs", "additional_context"],
            max_context_length=8000
        )
        
        # Drafting template
        self.templates[TaskType.DRAFTING] = PromptTemplate(
            task_type=TaskType.DRAFTING,
            system_prompt="""You are a skilled test automation engineer specializing in Playwright TypeScript test creation.
Your role is to write clean, maintainable, and reliable test code.

Key principles:
- Use semantic selectors (getByRole, getByLabel, getByTestId) over CSS selectors
- Implement proper waits using expect(locator).toBeVisible() instead of arbitrary timeouts
- Follow page object patterns for maintainability
- Write clear, descriptive test names and comments
- Handle edge cases and error scenarios appropriately

Always generate production-ready test code that follows best practices.""",
            user_template="""Please generate a Playwright TypeScript test based on this test plan:

**Test Case:** {test_case_name}

**Test Steps:**
{test_steps}

**Expected Results:**
{expected_results}

**Page URL:** {page_url}

**Selectors to Use:**
{selectors}

**Additional Requirements:**
{requirements}

Please generate:
1. Complete test file with proper imports
2. Page object classes if needed
3. Test data setup if required
4. Proper error handling and assertions
5. Comments explaining complex logic

Follow these selector preferences:
1. getByRole() for interactive elements
2. getByLabel() for form fields
3. getByTestId() for elements with test IDs
4. CSS selectors only when necessary (with justifying comments)""",
            context_keys=["test_case_name", "test_steps", "expected_results", 
                         "page_url", "selectors", "requirements"],
            max_context_length=5000
        )
        
        # Analysis template
        self.templates[TaskType.ANALYSIS] = PromptTemplate(
            task_type=TaskType.ANALYSIS,
            system_prompt="""You are an expert QA analyst specializing in test suite health and optimization.
Your role is to analyze test patterns, identify issues, and suggest improvements.

Key responsibilities:
- Analyze test code for patterns and anti-patterns
- Identify flaky tests and stability issues
- Suggest refactoring opportunities
- Evaluate selector quality and maintainability
- Recommend test suite optimizations

Provide actionable insights that improve test reliability and maintainability.""",
            user_template="""Please analyze the following test suite and provide recommendations:

**Analysis Type:** {analysis_type}

**Test Files:**
{test_files}

**Test Results History:**
{test_results}

**Specific Focus Areas:**
{focus_areas}

**Current Issues:**
{current_issues}

Please provide:
1. Overall test suite health assessment
2. Identified patterns and anti-patterns
3. Specific recommendations for improvement
4. Priority ranking of suggested changes
5. Estimated impact of each recommendation

Focus on practical improvements that can be implemented incrementally.""",
            context_keys=["analysis_type", "test_files", "test_results", 
                         "focus_areas", "current_issues"],
            max_context_length=7000
        )
        
        # Generation template (for general code generation)
        self.templates[TaskType.GENERATION] = PromptTemplate(
            task_type=TaskType.GENERATION,
            system_prompt="""You are a test automation engineer focused on generating high-quality test code and utilities.
Your role is to create reusable, maintainable code components for test automation.

Key principles:
- Write clean, well-documented code
- Follow TypeScript and Playwright best practices
- Create reusable utilities and helpers
- Implement proper error handling
- Consider maintainability and extensibility

Generate production-ready code that can be easily integrated into existing test suites.""",
            user_template="""Please generate the requested code component:

**Component Type:** {component_type}

**Requirements:**
{requirements}

**Specifications:**
{specifications}

**Integration Context:**
{integration_context}

**Constraints:**
{constraints}

Please provide:
1. Complete, working code implementation
2. Proper TypeScript types and interfaces
3. Comprehensive error handling
4. Usage examples and documentation
5. Unit tests if applicable

Ensure the code follows best practices and is ready for production use.""",
            context_keys=["component_type", "requirements", "specifications", 
                         "integration_context", "constraints"],
            max_context_length=6000
        )
    
    def get_template(self, task_type: TaskType) -> PromptTemplate:
        """
        Get the template for a specific task type.
        
        Args:
            task_type: The type of task
            
        Returns:
            PromptTemplate for the task type
            
        Raises:
            KeyError: If no template exists for the task type
        """
        if task_type not in self.templates:
            raise KeyError(f"No template found for task type: {task_type}")
        
        return self.templates[task_type]
    
    def register_template(self, template: PromptTemplate) -> None:
        """
        Register a custom template for a task type.
        
        Args:
            template: The template to register
        """
        self.templates[template.task_type] = template
    
    def list_available_templates(self) -> List[TaskType]:
        """Get list of available template task types."""
        return list(self.templates.keys())
    
    def format_prompt(
        self, 
        task_type: TaskType, 
        context: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Format a prompt for the given task type and context.
        
        Args:
            task_type: The type of task
            context: Context variables for the template
            
        Returns:
            Formatted messages in OpenAI format
        """
        template = self.get_template(task_type)
        return template.format(context)