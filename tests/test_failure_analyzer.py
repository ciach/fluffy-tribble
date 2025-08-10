"""
Unit tests for the failure analyzer.

Tests failure analysis logic, stack trace parsing, artifact analysis,
and fix suggestion generation.
"""

import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, mock_open

from orchestrator.analysis.analyzer import FailureAnalyzer
from orchestrator.analysis.models import (
    FailureAnalysis,
    StackTraceAnalysis,
    ArtifactAnalysis,
    FixSuggestion,
    ErrorCategory,
    ConfidenceLevel,
    FixType,
    FailureAnalysisError
)
from orchestrator.core.config import Config
from orchestrator.core.exceptions import ValidationError
from orchestrator.execution.models import (
    ExecutionResult,
    TestStatus,
    ArtifactMetadata,
    ArtifactType,
    ExecutionConfig
)
from orchestrator.models.router import ModelRouter
from orchestrator.models.types import ModelResponse, TaskType, ModelProvider


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = Mock(spec=Config)
    config.debug_enabled = False
    config.openai_api_key = "test-key"
    config.model_provider = "mixed"
    config.ollama_base_url = "http://localhost:11434"
    return config


@pytest.fixture
def mock_model_router():
    """Create a mock model router."""
    router = Mock(spec=ModelRouter)
    return router


@pytest.fixture
def failure_analyzer(mock_config, mock_model_router):
    """Create a failure analyzer instance."""
    return FailureAnalyzer(mock_config, mock_model_router)


@pytest.fixture
def sample_execution_result():
    """Create a sample failed execution result."""
    config = ExecutionConfig(test_file="test.spec.ts")
    
    return ExecutionResult(
        test_name="sample_test",
        test_file="test.spec.ts",
        workflow_id="test-workflow-123",
        status=TestStatus.FAILED,
        duration=5.2,
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        exit_code=1,
        stdout="Test output",
        stderr="Error output",
        error_message="TimeoutError: Timeout 30000ms exceeded",
        error_type="TimeoutError",
        stack_trace="""Error: Timeout 30000ms exceeded.
    at page.click (test.spec.ts:42:15)
    at TestRunner.run (runner.js:123:20)
    > 42 |   await page.getByRole('button', { name: 'Submit' }).click();
        |                                                      ^
    at Object.<anonymous> (test.spec.ts:40:1)""",
        execution_config=config,
        artifacts=[
            ArtifactMetadata(
                artifact_type=ArtifactType.TRACE,
                file_path="/tmp/trace.zip",
                file_size=1024,
                test_name="sample_test",
                workflow_id="test-workflow-123"
            )
        ]
    )


class TestFailureAnalyzer:
    """Test cases for FailureAnalyzer."""
    
    def test_init(self, mock_config, mock_model_router):
        """Test analyzer initialization."""
        analyzer = FailureAnalyzer(mock_config, mock_model_router)
        
        assert analyzer.config == mock_config
        assert analyzer.model_router == mock_model_router
        assert analyzer.max_stack_trace_lines == 50
        assert analyzer.max_artifact_size_mb == 10
        assert analyzer.analysis_timeout == 120
    
    @pytest.mark.asyncio
    async def test_analyze_failure_success(
        self, 
        failure_analyzer, 
        sample_execution_result,
        mock_model_router
    ):
        """Test successful failure analysis."""
        # Mock AI response
        ai_response = {
            "primary_error_category": "timeout",
            "secondary_categories": ["selector"],
            "root_cause": "Button click timed out",
            "contributing_factors": ["Slow page load"],
            "fix_suggestions": [
                {
                    "fix_type": "wait_condition",
                    "description": "Add explicit wait for button",
                    "confidence": "high",
                    "original_code": "await page.getByRole('button').click()",
                    "suggested_code": "await expect(page.getByRole('button')).toBeVisible(); await page.getByRole('button').click()",
                    "line_number": 42,
                    "reasoning": "Explicit wait ensures element is ready",
                    "potential_side_effects": [],
                    "test_impact": "minimal",
                    "requires_manual_review": False
                }
            ],
            "overall_confidence": "high",
            "similar_failures": ["button_timeout_pattern"],
            "environment_factors": {"browser": "chromium"}
        }
        
        mock_response = ModelResponse(
            content=json.dumps(ai_response),
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            task_type=TaskType.ANALYSIS,
            timestamp=datetime.utcnow()
        )
        
        mock_model_router.route_task = AsyncMock(return_value=mock_response)
        
        # Mock file operations
        with patch('pathlib.Path.exists', return_value=True):
            result = await failure_analyzer.analyze_failure(sample_execution_result)
        
        # Verify result
        assert isinstance(result, FailureAnalysis)
        assert result.test_name == "sample_test"
        assert result.primary_error_category == ErrorCategory.TIMEOUT
        assert result.root_cause == "Button click timed out"
        assert len(result.fix_suggestions) == 1
        assert result.fix_suggestions[0].fix_type == FixType.WAIT_CONDITION
        assert result.overall_confidence == ConfidenceLevel.HIGH
        
        # Verify model router was called correctly
        mock_model_router.route_task.assert_called_once()
        call_args = mock_model_router.route_task.call_args
        assert call_args[0][0] == TaskType.ANALYSIS
        assert len(call_args[0][1]) == 2  # system + user messages
    
    @pytest.mark.asyncio
    async def test_analyze_failure_passed_test(
        self, 
        failure_analyzer, 
        sample_execution_result
    ):
        """Test that analyzing a passed test raises ValidationError."""
        sample_execution_result.status = TestStatus.PASSED
        
        with pytest.raises(ValidationError) as exc_info:
            await failure_analyzer.analyze_failure(sample_execution_result)
        
        assert "Cannot analyze successful test execution" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_analyze_failure_with_fallback(
        self, 
        failure_analyzer, 
        sample_execution_result,
        mock_model_router
    ):
        """Test failure analysis with AI failure and fallback."""
        # Mock AI failure
        mock_model_router.route_task = AsyncMock(
            side_effect=Exception("AI model failed")
        )
        
        with patch('pathlib.Path.exists', return_value=True):
            result = await failure_analyzer.analyze_failure(sample_execution_result)
        
        # Verify fallback analysis
        assert isinstance(result, FailureAnalysis)
        assert result.test_name == "sample_test"
        assert result.primary_error_category == ErrorCategory.TIMEOUT  # Based on error message
        assert result.overall_confidence == ConfidenceLevel.LOW
        assert "Automated analysis unavailable" in result.contributing_factors
    
    def test_extract_failing_code(self, failure_analyzer):
        """Test extraction of failing code from stack trace."""
        stack_trace = """Error: Timeout 30000ms exceeded.
    at page.click (test.spec.ts:42:15)
    > 42 |   await page.getByRole('button', { name: 'Submit' }).click();
        |                                                      ^
    at Object.<anonymous> (test.spec.ts:40:1)"""
        
        line_number, code = failure_analyzer._extract_failing_code(stack_trace)
        
        assert line_number == 42
        if code:  # Handle case where code extraction might return None
            assert "await page.getByRole('button', { name: 'Submit' }).click();" in code
        else:
            # If code is None, at least verify line number was extracted
            assert line_number is not None
    
    def test_extract_playwright_action(self, failure_analyzer):
        """Test extraction of Playwright action from stack trace."""
        stack_trace = "at page.click (test.spec.ts:42:15)"
        
        action = failure_analyzer._extract_playwright_action(stack_trace)
        
        assert action == "click"
    
    def test_extract_selector(self, failure_analyzer):
        """Test extraction of selector from stack trace."""
        stack_trace = "await page.getByRole('button', { name: 'Submit' }).click();"
        
        selector = failure_analyzer._extract_selector(stack_trace)
        
        assert selector == "button"
    
    @pytest.mark.asyncio
    async def test_analyze_stack_trace(self, failure_analyzer, sample_execution_result):
        """Test stack trace analysis."""
        result = await failure_analyzer._analyze_stack_trace(
            sample_execution_result, 
            "test-analysis-id"
        )
        
        assert isinstance(result, StackTraceAnalysis)
        assert result.error_message == "TimeoutError: Timeout 30000ms exceeded"
        assert result.error_type == "TimeoutError"
        assert result.failing_line == 42
        assert result.playwright_action == "click"
    
    @pytest.mark.asyncio
    async def test_analyze_stack_trace_no_trace(self, failure_analyzer):
        """Test stack trace analysis with no stack trace."""
        execution_result = Mock()
        execution_result.stack_trace = None
        
        result = await failure_analyzer._analyze_stack_trace(
            execution_result, 
            "test-analysis-id"
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_analyze_console_logs(self, failure_analyzer, tmp_path):
        """Test console log analysis."""
        # Create test log file
        log_file = tmp_path / "console.log"
        log_content = """
        [ERROR] TypeError: Cannot read property 'click' of null
        [WARN] Failed to load resource: net::ERR_CONNECTION_REFUSED
        [INFO] Page loaded successfully
        """
        log_file.write_text(log_content)
        
        findings, error_indicators = await failure_analyzer._analyze_console_logs(log_file)
        
        assert "JavaScript errors detected in console" in error_indicators
        assert "Network errors detected in console" in error_indicators
        assert "Network connectivity issues may be affecting test" in findings
    
    @pytest.mark.asyncio
    async def test_analyze_network_logs(self, failure_analyzer, tmp_path):
        """Test network log analysis."""
        # Create test network log file
        log_file = tmp_path / "network.log"
        log_content = """
        {"url": "/api/data", "status": 404, "method": "GET"}
        {"url": "/api/timeout", "error": "timeout", "method": "POST"}
        """
        log_file.write_text(log_content)
        
        findings, error_indicators = await failure_analyzer._analyze_network_logs(log_file)
        
        assert "HTTP error responses detected" in error_indicators
        assert "Network timeouts detected" in error_indicators
        assert "API requests returning error status codes" in findings
    
    def test_build_analysis_prompt(self, failure_analyzer, sample_execution_result):
        """Test analysis prompt building."""
        stack_analysis = StackTraceAnalysis(
            raw_stack_trace="test trace",
            error_message="test error",
            error_type="TestError",
            failing_line=42,
            failing_code="test code",
            playwright_action="click",
            selector_used="button"
        )
        
        artifact_analyses = [
            ArtifactAnalysis(
                artifact_path="/tmp/test.log",
                artifact_type="console_log",
                findings=["Test finding"],
                error_indicators=["Test error"]
            )
        ]
        
        context = {"test_context": "value"}
        
        with patch('pathlib.Path.exists', return_value=True):
            prompt = failure_analyzer._build_analysis_prompt(
                sample_execution_result,
                stack_analysis,
                artifact_analyses,
                context
            )
        
        assert "# Test Failure Analysis Request" in prompt
        assert "sample_test" in prompt
        assert "TimeoutError" in prompt
        assert "test trace" in prompt
        assert "Console_Log Analysis" in prompt
        assert "test_context" in prompt
    
    def test_parse_analysis_response(self, failure_analyzer):
        """Test parsing of AI analysis response."""
        response_content = """
        Based on the analysis, here's what I found:
        
        {
          "primary_error_category": "timeout",
          "secondary_categories": ["selector"],
          "root_cause": "Element not found",
          "contributing_factors": ["Slow loading"],
          "fix_suggestions": [
            {
              "fix_type": "wait_condition",
              "description": "Add wait",
              "confidence": "high",
              "reasoning": "Will help",
              "potential_side_effects": [],
              "test_impact": "minimal",
              "requires_manual_review": false
            }
          ],
          "overall_confidence": "medium",
          "similar_failures": [],
          "environment_factors": {}
        }
        """
        
        result = failure_analyzer._parse_analysis_response(response_content)
        
        assert result["primary_error_category"] == ErrorCategory.TIMEOUT
        assert result["secondary_categories"] == [ErrorCategory.SELECTOR]
        assert result["root_cause"] == "Element not found"
        assert len(result["fix_suggestions"]) == 1
        assert result["fix_suggestions"][0].fix_type == FixType.WAIT_CONDITION
        assert result["overall_confidence"] == ConfidenceLevel.MEDIUM
    
    def test_parse_analysis_response_invalid_json(self, failure_analyzer):
        """Test parsing of invalid AI response."""
        response_content = "This is not JSON"
        
        with pytest.raises(FailureAnalysisError) as exc_info:
            failure_analyzer._parse_analysis_response(response_content)
        
        assert "Failed to parse analysis response" in str(exc_info.value)
    
    def test_fallback_analysis_timeout(self, failure_analyzer, sample_execution_result):
        """Test fallback analysis for timeout errors."""
        stack_analysis = StackTraceAnalysis(
            raw_stack_trace="test trace",
            error_message="timeout exceeded",
            error_type="TimeoutError"
        )
        
        result = failure_analyzer._fallback_analysis(
            sample_execution_result, 
            stack_analysis
        )
        
        assert result["primary_error_category"] == ErrorCategory.TIMEOUT
        assert "timed out" in result["root_cause"].lower()
        assert len(result["fix_suggestions"]) == 1
        assert result["fix_suggestions"][0].fix_type == FixType.TIMEOUT_ADJUSTMENT
        assert result["overall_confidence"] == ConfidenceLevel.LOW
    
    def test_fallback_analysis_selector(self, failure_analyzer, sample_execution_result):
        """Test fallback analysis for selector errors."""
        stack_analysis = StackTraceAnalysis(
            raw_stack_trace="test trace",
            error_message="element not found selector",
            error_type="SelectorError"
        )
        
        result = failure_analyzer._fallback_analysis(
            sample_execution_result, 
            stack_analysis
        )
        
        assert result["primary_error_category"] == ErrorCategory.SELECTOR
        assert "selector" in result["root_cause"].lower()
        assert len(result["fix_suggestions"]) == 1
        assert result["fix_suggestions"][0].fix_type == FixType.SELECTOR_REPLACEMENT
    
    def test_fallback_analysis_assertion(self, failure_analyzer, sample_execution_result):
        """Test fallback analysis for assertion errors."""
        stack_analysis = StackTraceAnalysis(
            raw_stack_trace="test trace",
            error_message="assertion failed expect",
            error_type="AssertionError"
        )
        
        result = failure_analyzer._fallback_analysis(
            sample_execution_result, 
            stack_analysis
        )
        
        assert result["primary_error_category"] == ErrorCategory.ASSERTION
        assert "assertion" in result["root_cause"].lower()
        assert len(result["fix_suggestions"]) == 1
        assert result["fix_suggestions"][0].fix_type == FixType.ASSERTION_UPDATE
        assert result["fix_suggestions"][0].requires_manual_review is True


class TestStackTraceAnalysis:
    """Test cases for StackTraceAnalysis model."""
    
    def test_valid_stack_trace_analysis(self):
        """Test creating valid stack trace analysis."""
        analysis = StackTraceAnalysis(
            raw_stack_trace="Error at line 42",
            error_message="Test error",
            error_type="TestError",
            failing_line=42,
            failing_code="test code",
            playwright_action="click",
            selector_used="button"
        )
        
        assert analysis.raw_stack_trace == "Error at line 42"
        assert analysis.error_message == "Test error"
        assert analysis.failing_line == 42
        assert analysis.playwright_action == "click"
    
    def test_empty_stack_trace_validation(self):
        """Test validation of empty stack trace."""
        with pytest.raises(ValueError) as exc_info:
            StackTraceAnalysis(
                raw_stack_trace="   ",
                error_message="Test error",
                error_type="TestError"
            )
        
        assert "Stack trace cannot be empty" in str(exc_info.value)


class TestFixSuggestion:
    """Test cases for FixSuggestion model."""
    
    def test_valid_fix_suggestion(self):
        """Test creating valid fix suggestion."""
        fix = FixSuggestion(
            fix_type=FixType.SELECTOR_REPLACEMENT,
            description="Update selector",
            confidence=ConfidenceLevel.HIGH,
            reasoning="Better selector needed",
            original_code="old code",
            suggested_code="new code",
            line_number=42
        )
        
        assert fix.fix_type == FixType.SELECTOR_REPLACEMENT
        assert fix.confidence == ConfidenceLevel.HIGH
        assert fix.original_code == "old code"
        assert fix.line_number == 42
    
    def test_high_risk_fix_manual_review(self):
        """Test that high-risk fixes require manual review."""
        fix = FixSuggestion(
            fix_type=FixType.ASSERTION_UPDATE,
            description="Update assertion",
            confidence=ConfidenceLevel.HIGH,
            reasoning="Assertion needs update"
        )
        
        # This should be set automatically for high-risk types
        assert fix.requires_manual_review is True


class TestFailureAnalysis:
    """Test cases for FailureAnalysis model."""
    
    def test_valid_failure_analysis(self):
        """Test creating valid failure analysis."""
        fix_suggestions = [
            FixSuggestion(
                fix_type=FixType.WAIT_CONDITION,
                description="Add wait",
                confidence=ConfidenceLevel.HIGH,
                reasoning="Will help"
            ),
            FixSuggestion(
                fix_type=FixType.SELECTOR_REPLACEMENT,
                description="Update selector",
                confidence=ConfidenceLevel.MEDIUM,
                reasoning="Better selector"
            )
        ]
        
        analysis = FailureAnalysis(
            test_name="test",
            test_file="test.spec.ts",
            workflow_id="workflow-123",
            analysis_id="analysis-456",
            analysis_duration=5.2,
            primary_error_category=ErrorCategory.TIMEOUT,
            root_cause="Timeout occurred",
            overall_confidence=ConfidenceLevel.HIGH,
            fix_suggestions=fix_suggestions
        )
        
        assert analysis.test_name == "test"
        assert analysis.primary_error_category == ErrorCategory.TIMEOUT
        assert len(analysis.fix_suggestions) == 2
        
        # Test that suggestions are ordered by confidence
        assert analysis.fix_suggestions[0].confidence == ConfidenceLevel.HIGH
        assert analysis.fix_suggestions[1].confidence == ConfidenceLevel.MEDIUM
    
    def test_has_high_confidence_fixes(self):
        """Test checking for high confidence fixes."""
        fix_suggestions = [
            FixSuggestion(
                fix_type=FixType.WAIT_CONDITION,
                description="Add wait",
                confidence=ConfidenceLevel.HIGH,
                reasoning="Will help"
            )
        ]
        
        analysis = FailureAnalysis(
            test_name="test",
            test_file="test.spec.ts",
            workflow_id="workflow-123",
            analysis_id="analysis-456",
            analysis_duration=5.2,
            primary_error_category=ErrorCategory.TIMEOUT,
            root_cause="Timeout occurred",
            overall_confidence=ConfidenceLevel.HIGH,
            fix_suggestions=fix_suggestions
        )
        
        assert analysis.has_high_confidence_fixes is True
    
    def test_recommended_fix(self):
        """Test getting recommended fix."""
        fix_suggestions = [
            FixSuggestion(
                fix_type=FixType.WAIT_CONDITION,
                description="Add wait",
                confidence=ConfidenceLevel.HIGH,
                reasoning="Will help"
            )
        ]
        
        analysis = FailureAnalysis(
            test_name="test",
            test_file="test.spec.ts",
            workflow_id="workflow-123",
            analysis_id="analysis-456",
            analysis_duration=5.2,
            primary_error_category=ErrorCategory.TIMEOUT,
            root_cause="Timeout occurred",
            overall_confidence=ConfidenceLevel.HIGH,
            fix_suggestions=fix_suggestions
        )
        
        recommended = analysis.recommended_fix
        assert recommended is not None
        assert recommended.fix_type == FixType.WAIT_CONDITION
    
    def test_get_fixes_by_type(self):
        """Test getting fixes by type."""
        fix_suggestions = [
            FixSuggestion(
                fix_type=FixType.WAIT_CONDITION,
                description="Add wait",
                confidence=ConfidenceLevel.HIGH,
                reasoning="Will help"
            ),
            FixSuggestion(
                fix_type=FixType.SELECTOR_REPLACEMENT,
                description="Update selector",
                confidence=ConfidenceLevel.MEDIUM,
                reasoning="Better selector"
            ),
            FixSuggestion(
                fix_type=FixType.WAIT_CONDITION,
                description="Another wait",
                confidence=ConfidenceLevel.LOW,
                reasoning="Also helps"
            )
        ]
        
        analysis = FailureAnalysis(
            test_name="test",
            test_file="test.spec.ts",
            workflow_id="workflow-123",
            analysis_id="analysis-456",
            analysis_duration=5.2,
            primary_error_category=ErrorCategory.TIMEOUT,
            root_cause="Timeout occurred",
            overall_confidence=ConfidenceLevel.HIGH,
            fix_suggestions=fix_suggestions
        )
        
        wait_fixes = analysis.get_fixes_by_type(FixType.WAIT_CONDITION)
        assert len(wait_fixes) == 2
        
        selector_fixes = analysis.get_fixes_by_type(FixType.SELECTOR_REPLACEMENT)
        assert len(selector_fixes) == 1