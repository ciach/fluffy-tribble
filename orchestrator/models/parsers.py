"""
Response parsing and validation utilities for model interactions.

Provides structured parsing of model responses for different task types.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

from .types import TaskType, ModelResponse

logger = logging.getLogger(__name__)


class ParsedResponseType(Enum):
    """Types of parsed responses."""
    TEST_PLAN = "test_plan"
    DEBUG_ANALYSIS = "debug_analysis"
    TEST_CODE = "test_code"
    ANALYSIS_REPORT = "analysis_report"
    CODE_COMPONENT = "code_component"


@dataclass
class ParsedResponse:
    """Structured representation of a parsed model response."""
    
    response_type: ParsedResponseType
    content: Dict[str, Any]
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    validation_errors: Optional[List[str]] = None
    
    @property
    def is_valid(self) -> bool:
        """Check if the parsed response is valid."""
        return not self.validation_errors or len(self.validation_errors) == 0


@dataclass
class TestPlan:
    """Structured test plan from planning tasks."""
    
    objectives: List[str]
    test_cases: List[Dict[str, Any]]
    data_requirements: List[str]
    coverage_analysis: str
    risk_assessment: str
    estimated_effort: Optional[str] = None
    priority: Optional[str] = None


@dataclass
class DebugAnalysis:
    """Structured debug analysis from debugging tasks."""
    
    root_cause: str
    error_category: str
    code_changes: List[Dict[str, str]]
    explanation: str
    prevention_suggestions: List[str]
    confidence_level: int
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if the analysis has high confidence."""
        return self.confidence_level >= 7


@dataclass
class TestCode:
    """Structured test code from drafting tasks."""
    
    test_file_content: str
    page_objects: Optional[List[str]] = None
    test_data: Optional[str] = None
    imports: Optional[List[str]] = None
    comments: Optional[List[str]] = None
    
    def extract_selectors(self) -> List[str]:
        """Extract selectors used in the test code."""
        selectors = []
        
        # Common Playwright selector patterns
        patterns = [
            r'getByRole\([\'"]([^\'"]+)[\'"]',
            r'getByLabel\([\'"]([^\'"]+)[\'"]',
            r'getByTestId\([\'"]([^\'"]+)[\'"]',
            r'getByText\([\'"]([^\'"]+)[\'"]',
            r'locator\([\'"]([^\'"]+)[\'"]',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, self.test_file_content)
            selectors.extend(matches)
        
        return list(set(selectors))  # Remove duplicates


class ResponseParser:
    """Parses and validates model responses for different task types."""
    
    def __init__(self):
        """Initialize the response parser."""
        self.parsers = {
            TaskType.PLANNING: self._parse_planning_response,
            TaskType.DEBUGGING: self._parse_debugging_response,
            TaskType.DRAFTING: self._parse_drafting_response,
            TaskType.ANALYSIS: self._parse_analysis_response,
            TaskType.GENERATION: self._parse_generation_response,
        }
    
    def parse_response(
        self, 
        model_response: ModelResponse
    ) -> ParsedResponse:
        """
        Parse a model response based on its task type.
        
        Args:
            model_response: The response from the model
            
        Returns:
            ParsedResponse with structured content
        """
        parser = self.parsers.get(model_response.task_type)
        if not parser:
            logger.warning(f"No parser available for task type: {model_response.task_type}")
            return ParsedResponse(
                response_type=ParsedResponseType.CODE_COMPONENT,
                content={"raw_content": model_response.content},
                validation_errors=["No parser available for task type"]
            )
        
        try:
            return parser(model_response.content)
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            return ParsedResponse(
                response_type=ParsedResponseType.CODE_COMPONENT,
                content={"raw_content": model_response.content},
                validation_errors=[f"Parsing failed: {str(e)}"]
            )
    
    def _parse_planning_response(self, content: str) -> ParsedResponse:
        """Parse a planning task response."""
        validation_errors = []
        
        # Extract sections using regex patterns
        objectives = self._extract_list_section(content, r"(?:Test Objectives|Objectives):(.*?)(?=\n\d+\.|$)", "objectives")
        test_cases = self._extract_test_cases(content)
        data_requirements = self._extract_list_section(content, r"(?:Test Data Requirements|Data Requirements):(.*?)(?=\n\d+\.|$)", "data requirements")
        coverage_analysis = self._extract_text_section(content, r"(?:Coverage Analysis):(.*?)(?=\n\d+\.|$)")
        risk_assessment = self._extract_text_section(content, r"(?:Risk Assessment):(.*?)(?=\n\d+\.|$)")
        
        # Validation
        if not objectives:
            validation_errors.append("No test objectives found")
        if not test_cases:
            validation_errors.append("No test cases found")
        
        test_plan = TestPlan(
            objectives=objectives,
            test_cases=test_cases,
            data_requirements=data_requirements,
            coverage_analysis=coverage_analysis,
            risk_assessment=risk_assessment
        )
        
        return ParsedResponse(
            response_type=ParsedResponseType.TEST_PLAN,
            content={"test_plan": test_plan.__dict__},
            validation_errors=validation_errors if validation_errors else None
        )
    
    def _parse_debugging_response(self, content: str) -> ParsedResponse:
        """Parse a debugging task response."""
        validation_errors = []
        
        # Extract debug analysis components
        root_cause = self._extract_text_section(content, r"(?:Root [Cc]ause|Root Cause Analysis):(.*?)(?=\n\d+\.|$)")
        code_changes = self._extract_code_changes(content)
        explanation = self._extract_text_section(content, r"(?:Explanation):(.*?)(?=\n\d+\.|$)")
        prevention_suggestions = self._extract_list_section(content, r"(?:Prevention|Suggestions):(.*?)(?=\n\d+\.|$)", "prevention suggestions")
        confidence_level = self._extract_confidence_level(content)
        
        # Determine error category
        error_category = self._categorize_error(content)
        
        # Validation
        if not root_cause:
            validation_errors.append("No root cause identified")
        if not code_changes:
            validation_errors.append("No code changes provided")
        if confidence_level < 1:
            validation_errors.append("No confidence level specified")
        
        debug_analysis = DebugAnalysis(
            root_cause=root_cause,
            error_category=error_category,
            code_changes=code_changes,
            explanation=explanation,
            prevention_suggestions=prevention_suggestions,
            confidence_level=confidence_level
        )
        
        return ParsedResponse(
            response_type=ParsedResponseType.DEBUG_ANALYSIS,
            content={"debug_analysis": debug_analysis.__dict__},
            validation_errors=validation_errors if validation_errors else None
        )
    
    def _parse_drafting_response(self, content: str) -> ParsedResponse:
        """Parse a drafting task response."""
        validation_errors = []
        
        # Extract test code components
        test_file_content = self._extract_code_block(content, "typescript") or self._extract_code_block(content, "ts")
        if not test_file_content:
            # Try to extract any code block
            test_file_content = self._extract_first_code_block(content)
        
        page_objects = self._extract_page_objects(content)
        test_data = self._extract_test_data(content)
        imports = self._extract_imports(test_file_content) if test_file_content else []
        
        # Validation
        if not test_file_content:
            validation_errors.append("No test code found")
        elif not self._validate_playwright_code(test_file_content):
            validation_errors.append("Invalid Playwright test code")
        
        test_code = TestCode(
            test_file_content=test_file_content or "",
            page_objects=page_objects,
            test_data=test_data,
            imports=imports
        )
        
        return ParsedResponse(
            response_type=ParsedResponseType.TEST_CODE,
            content={"test_code": test_code.__dict__},
            validation_errors=validation_errors if validation_errors else None
        )
    
    def _parse_analysis_response(self, content: str) -> ParsedResponse:
        """Parse an analysis task response."""
        validation_errors = []
        
        # Extract analysis components
        health_assessment = self._extract_text_section(content, r"(?:Health Assessment|Overall.*Assessment):(.*?)(?=\n\d+\.|$)")
        patterns = self._extract_list_section(content, r"(?:Patterns|Anti-patterns):(.*?)(?=\n\d+\.|$)", "patterns")
        recommendations = self._extract_list_section(content, r"(?:Recommendations):(.*?)(?=\n\d+\.|$)", "recommendations")
        priorities = self._extract_list_section(content, r"(?:Priority|Priorities):(.*?)(?=\n\d+\.|$)", "priorities")
        
        # Validation
        if not health_assessment:
            validation_errors.append("No health assessment found")
        if not recommendations:
            validation_errors.append("No recommendations provided")
        
        analysis_report = {
            "health_assessment": health_assessment,
            "patterns": patterns,
            "recommendations": recommendations,
            "priorities": priorities
        }
        
        return ParsedResponse(
            response_type=ParsedResponseType.ANALYSIS_REPORT,
            content={"analysis_report": analysis_report},
            validation_errors=validation_errors if validation_errors else None
        )
    
    def _parse_generation_response(self, content: str) -> ParsedResponse:
        """Parse a generation task response."""
        validation_errors = []
        
        # Extract generated code
        code_content = self._extract_code_block(content, "typescript") or self._extract_first_code_block(content)
        documentation = self._extract_documentation(content)
        usage_examples = self._extract_usage_examples(content)
        
        # Validation
        if not code_content:
            validation_errors.append("No code content found")
        
        code_component = {
            "code_content": code_content or "",
            "documentation": documentation,
            "usage_examples": usage_examples
        }
        
        return ParsedResponse(
            response_type=ParsedResponseType.CODE_COMPONENT,
            content={"code_component": code_component},
            validation_errors=validation_errors if validation_errors else None
        )
    
    def _extract_list_section(self, content: str, pattern: str, section_name: str) -> List[str]:
        """Extract a list section from content using regex."""
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if not match:
            return []
        
        section_text = match.group(1).strip()
        # Extract list items (numbered or bulleted)
        items = re.findall(r'(?:^\d+\.|^[-*])\s*(.+)', section_text, re.MULTILINE)
        return [item.strip() for item in items if item.strip()]
    
    def _extract_text_section(self, content: str, pattern: str) -> str:
        """Extract a text section from content using regex."""
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if not match:
            return ""
        
        return match.group(1).strip()
    
    def _extract_test_cases(self, content: str) -> List[Dict[str, Any]]:
        """Extract test cases from planning response."""
        test_cases = []
        
        # Look for test case patterns
        case_pattern = r'(?:Test Case|Case)\s*\d*:?\s*(.+?)(?=(?:Test Case|Case)\s*\d*:|$)'
        matches = re.findall(case_pattern, content, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            case_text = match.strip()
            if case_text:
                # Extract steps and expected results
                steps = self._extract_list_section(case_text, r'(?:Steps|Test Steps):(.*?)(?=Expected|$)', "steps")
                expected = self._extract_text_section(case_text, r'(?:Expected|Expected Results?):(.*?)$')
                
                test_cases.append({
                    "description": case_text.split('\n')[0].strip(),
                    "steps": steps,
                    "expected_result": expected
                })
        
        return test_cases
    
    def _extract_code_changes(self, content: str) -> List[Dict[str, str]]:
        """Extract code changes from debugging response."""
        changes = []
        
        # Look for code blocks or specific change patterns
        code_blocks = re.findall(r'```(?:typescript|ts|javascript|js)?\n(.*?)\n```', content, re.DOTALL)
        
        for i, code in enumerate(code_blocks):
            changes.append({
                "type": "code_replacement",
                "content": code.strip(),
                "description": f"Code change {i + 1}"
            })
        
        return changes
    
    def _extract_confidence_level(self, content: str) -> int:
        """Extract confidence level from debugging response."""
        confidence_match = re.search(r'confidence.*?(\d+)', content, re.IGNORECASE)
        if confidence_match:
            return int(confidence_match.group(1))
        return 0
    
    def _categorize_error(self, content: str) -> str:
        """Categorize the error type based on content."""
        content_lower = content.lower()
        
        if any(term in content_lower for term in ['selector', 'locator', 'element']):
            return 'selector'
        elif any(term in content_lower for term in ['timeout', 'wait', 'timing']):
            return 'timing'
        elif any(term in content_lower for term in ['network', 'request', 'response']):
            return 'network'
        elif any(term in content_lower for term in ['assertion', 'expect', 'should']):
            return 'assertion'
        else:
            return 'other'
    
    def _extract_code_block(self, content: str, language: str) -> Optional[str]:
        """Extract code block for specific language."""
        pattern = f'```{language}\\n(.*?)\\n```'
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else None
    
    def _extract_first_code_block(self, content: str) -> Optional[str]:
        """Extract the first code block found."""
        pattern = r'```(?:\w+)?\n(.*?)\n```'
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else None
    
    def _extract_page_objects(self, content: str) -> Optional[List[str]]:
        """Extract page object class names from content."""
        # Look for class definitions
        class_pattern = r'class\s+(\w+Page)\s*{'
        matches = re.findall(class_pattern, content)
        return matches if matches else None
    
    def _extract_test_data(self, content: str) -> Optional[str]:
        """Extract test data from content."""
        data_pattern = r'(?:test data|test setup):(.*?)(?=\n\n|$)'
        match = re.search(data_pattern, content, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None
    
    def _extract_imports(self, code: str) -> List[str]:
        """Extract import statements from code."""
        if not code:
            return []
        
        import_pattern = r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]'
        matches = re.findall(import_pattern, code)
        return matches
    
    def _validate_playwright_code(self, code: str) -> bool:
        """Validate that code contains Playwright patterns."""
        playwright_patterns = [
            r'test\(',
            r'expect\(',
            r'page\.',
            r'getBy',
            r'locator\(',
        ]
        
        return any(re.search(pattern, code) for pattern in playwright_patterns)
    
    def _extract_documentation(self, content: str) -> Optional[str]:
        """Extract documentation from generation response."""
        doc_pattern = r'(?:documentation|usage|description):(.*?)(?=\n\n|```|$)'
        match = re.search(doc_pattern, content, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None
    
    def _extract_usage_examples(self, content: str) -> Optional[List[str]]:
        """Extract usage examples from generation response."""
        example_pattern = r'(?:example|usage).*?```(?:\w+)?\n(.*?)\n```'
        matches = re.findall(example_pattern, content, re.DOTALL | re.IGNORECASE)
        return [match.strip() for match in matches] if matches else None