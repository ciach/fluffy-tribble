"""
Failure analysis engine using OpenAI model.

Analyzes test failures by examining stack traces, artifacts, and execution context
to identify root causes and suggest fixes.
"""

import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from ..core.config import Config
from ..core.exceptions import ModelError, ValidationError
from ..execution.models import ExecutionResult, TestStatus, ArtifactType
from ..models.router import ModelRouter
from ..models.types import TaskType
from .models import (
    FailureAnalysis,
    StackTraceAnalysis,
    ArtifactAnalysis,
    FixSuggestion,
    ErrorCategory,
    ConfidenceLevel,
    FixType,
    FailureAnalysisError
)

logger = logging.getLogger(__name__)


class FailureAnalyzer:
    """
    Analyzes test failures and suggests fixes using AI models.
    
    Uses OpenAI models for complex failure analysis and root cause identification.
    Processes stack traces, artifacts, and execution context to provide actionable
    fix suggestions.
    """
    
    def __init__(self, config: Config, model_router: ModelRouter):
        """Initialize the failure analyzer."""
        self.config = config
        self.model_router = model_router
        
        # Analysis configuration
        self.max_stack_trace_lines = 50
        self.max_artifact_size_mb = 10
        self.analysis_timeout = 120  # seconds
        
        logger.info("Failure analyzer initialized")
    
    async def analyze_failure(
        self, 
        execution_result: ExecutionResult,
        context: Optional[Dict[str, Any]] = None
    ) -> FailureAnalysis:
        """
        Analyze a test failure and provide fix suggestions.
        
        Args:
            execution_result: The failed test execution result
            context: Optional additional context for analysis
            
        Returns:
            FailureAnalysis with root cause and fix suggestions
            
        Raises:
            FailureAnalysisError: If analysis fails
            ValidationError: If input validation fails
        """
        if execution_result.status == TestStatus.PASSED:
            raise ValidationError(
                "Cannot analyze successful test execution",
                validation_type="input"
            )
        
        analysis_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        logger.info(
            f"Starting failure analysis for {execution_result.test_name} "
            f"(analysis_id: {analysis_id})"
        )
        
        try:
            # Parse stack trace
            stack_trace_analysis = await self._analyze_stack_trace(
                execution_result, analysis_id
            )
            
            # Analyze artifacts
            artifact_analyses = await self._analyze_artifacts(
                execution_result, analysis_id
            )
            
            # Perform comprehensive analysis
            analysis_result = await self._perform_comprehensive_analysis(
                execution_result,
                stack_trace_analysis,
                artifact_analyses,
                context or {}
            )
            
            # Calculate analysis duration
            analysis_duration = (datetime.utcnow() - start_time).total_seconds()
            
            # Create failure analysis
            failure_analysis = FailureAnalysis(
                test_name=execution_result.test_name,
                test_file=execution_result.test_file,
                workflow_id=execution_result.workflow_id,
                analysis_id=analysis_id,
                analysis_duration=analysis_duration,
                stack_trace_analysis=stack_trace_analysis,
                artifact_analyses=artifact_analyses,
                **analysis_result
            )
            
            logger.info(
                f"Failure analysis completed for {execution_result.test_name} "
                f"in {analysis_duration:.2f}s with {len(failure_analysis.fix_suggestions)} suggestions"
            )
            
            return failure_analysis
            
        except Exception as e:
            analysis_duration = (datetime.utcnow() - start_time).total_seconds()
            logger.error(
                f"Failure analysis failed for {execution_result.test_name} "
                f"after {analysis_duration:.2f}s: {e}"
            )
            raise FailureAnalysisError(
                f"Analysis failed: {e}",
                test_name=execution_result.test_name,
                analysis_id=analysis_id
            )
    
    async def _analyze_stack_trace(
        self, 
        execution_result: ExecutionResult,
        analysis_id: str
    ) -> Optional[StackTraceAnalysis]:
        """Analyze the stack trace from the execution result."""
        if not execution_result.stack_trace:
            logger.debug(f"No stack trace available for {execution_result.test_name}")
            return None
        
        try:
            # Parse stack trace components
            error_message = execution_result.error_message or "Unknown error"
            error_type = execution_result.error_type or "UnknownError"
            
            # Extract key information from stack trace
            failing_line, failing_code = self._extract_failing_code(
                execution_result.stack_trace
            )
            playwright_action = self._extract_playwright_action(
                execution_result.stack_trace
            )
            selector_used = self._extract_selector(execution_result.stack_trace)
            
            return StackTraceAnalysis(
                raw_stack_trace=execution_result.stack_trace,
                error_message=error_message,
                error_type=error_type,
                failing_line=failing_line,
                failing_code=failing_code,
                playwright_action=playwright_action,
                selector_used=selector_used
            )
            
        except Exception as e:
            logger.warning(f"Failed to analyze stack trace: {e}")
            return None
    
    def _extract_failing_code(self, stack_trace: str) -> Tuple[Optional[int], Optional[str]]:
        """Extract the failing line number and code from stack trace."""
        try:
            # Look for patterns like "at /path/to/test.spec.ts:42:15"
            line_pattern = r"at.*?:(\d+):\d+"
            match = re.search(line_pattern, stack_trace)
            
            if match:
                line_number = int(match.group(1))
                
                # Try to extract the actual code line
                lines = stack_trace.split('\n')
                for line in lines:
                    if f":{line_number}:" in line and ">" in line:
                        # Extract code after the ">" marker
                        code_match = re.search(r'>\s*(.+)', line)
                        if code_match:
                            return line_number, code_match.group(1).strip()
                
                return line_number, None
            
            return None, None
            
        except Exception as e:
            logger.debug(f"Failed to extract failing code: {e}")
            return None, None
    
    def _extract_playwright_action(self, stack_trace: str) -> Optional[str]:
        """Extract the Playwright action that failed."""
        try:
            # Look for common Playwright actions
            actions = [
                'click', 'fill', 'type', 'press', 'hover', 'focus',
                'selectOption', 'check', 'uncheck', 'setInputFiles',
                'waitForSelector', 'waitForVisible', 'waitForHidden',
                'goto', 'goBack', 'goForward', 'reload'
            ]
            
            for action in actions:
                if f".{action}(" in stack_trace:
                    return action
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to extract Playwright action: {e}")
            return None
    
    def _extract_selector(self, stack_trace: str) -> Optional[str]:
        """Extract the selector that was used in the failing action."""
        try:
            # Look for common selector patterns
            selector_patterns = [
                r"getByRole\(['\"]([^'\"]+)['\"]",
                r"getByLabel\(['\"]([^'\"]+)['\"]",
                r"getByTestId\(['\"]([^'\"]+)['\"]",
                r"getByText\(['\"]([^'\"]+)['\"]",
                r"locator\(['\"]([^'\"]+)['\"]",
                r"querySelector\(['\"]([^'\"]+)['\"]"
            ]
            
            for pattern in selector_patterns:
                match = re.search(pattern, stack_trace)
                if match:
                    return match.group(1)
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to extract selector: {e}")
            return None
    
    async def _analyze_artifacts(
        self, 
        execution_result: ExecutionResult,
        analysis_id: str
    ) -> List[ArtifactAnalysis]:
        """Analyze test artifacts for additional failure context."""
        analyses = []
        
        for artifact in execution_result.artifacts:
            try:
                # Skip large artifacts
                if artifact.file_size > self.max_artifact_size_mb * 1024 * 1024:
                    logger.debug(
                        f"Skipping large artifact {artifact.file_path} "
                        f"({artifact.file_size / 1024 / 1024:.1f}MB)"
                    )
                    continue
                
                analysis = await self._analyze_single_artifact(artifact, analysis_id)
                if analysis:
                    analyses.append(analysis)
                    
            except Exception as e:
                logger.warning(f"Failed to analyze artifact {artifact.file_path}: {e}")
                continue
        
        return analyses
    
    async def _analyze_single_artifact(
        self, 
        artifact_metadata,
        analysis_id: str
    ) -> Optional[ArtifactAnalysis]:
        """Analyze a single test artifact."""
        try:
            artifact_path = Path(artifact_metadata.file_path)
            
            if not artifact_path.exists():
                logger.warning(f"Artifact file not found: {artifact_path}")
                return None
            
            findings = []
            error_indicators = []
            relevant_sections = []
            
            # Analyze based on artifact type
            if artifact_metadata.artifact_type == ArtifactType.CONSOLE_LOG:
                findings, error_indicators = await self._analyze_console_logs(artifact_path)
            elif artifact_metadata.artifact_type == ArtifactType.NETWORK_LOG:
                findings, error_indicators = await self._analyze_network_logs(artifact_path)
            elif artifact_metadata.artifact_type == ArtifactType.TRACE:
                findings, error_indicators = await self._analyze_trace_file(artifact_path)
            
            return ArtifactAnalysis(
                artifact_path=str(artifact_path),
                artifact_type=artifact_metadata.artifact_type.value,
                findings=findings,
                error_indicators=error_indicators,
                relevant_sections=relevant_sections
            )
            
        except Exception as e:
            logger.debug(f"Failed to analyze artifact: {e}")
            return None
    
    async def _analyze_console_logs(self, log_path: Path) -> Tuple[List[str], List[str]]:
        """Analyze console logs for errors and warnings."""
        findings = []
        error_indicators = []
        
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for JavaScript errors
            if 'Error:' in content or 'TypeError:' in content:
                error_indicators.append("JavaScript errors detected in console")
            
            # Look for network errors
            if 'Failed to load resource' in content or 'net::ERR_' in content:
                error_indicators.append("Network errors detected in console")
                findings.append("Network connectivity issues may be affecting test")
            
            # Look for warnings
            if 'Warning:' in content:
                findings.append("Console warnings detected")
            
        except Exception as e:
            logger.debug(f"Failed to analyze console logs: {e}")
        
        return findings, error_indicators
    
    async def _analyze_network_logs(self, log_path: Path) -> Tuple[List[str], List[str]]:
        """Analyze network logs for request failures."""
        findings = []
        error_indicators = []
        
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for HTTP error status codes
            if re.search(r'status":\s*[45]\d\d', content):
                error_indicators.append("HTTP error responses detected")
                findings.append("API requests returning error status codes")
            
            # Look for timeout errors
            if 'timeout' in content.lower():
                error_indicators.append("Network timeouts detected")
                findings.append("Network requests timing out")
            
        except Exception as e:
            logger.debug(f"Failed to analyze network logs: {e}")
        
        return findings, error_indicators
    
    async def _analyze_trace_file(self, trace_path: Path) -> Tuple[List[str], List[str]]:
        """Analyze Playwright trace file for failure context."""
        findings = []
        error_indicators = []
        
        try:
            # For now, just check if trace file exists and has reasonable size
            if trace_path.stat().st_size > 0:
                findings.append("Playwright trace captured successfully")
            else:
                error_indicators.append("Empty trace file - may indicate browser crash")
            
        except Exception as e:
            logger.debug(f"Failed to analyze trace file: {e}")
        
        return findings, error_indicators
    
    async def _perform_comprehensive_analysis(
        self,
        execution_result: ExecutionResult,
        stack_trace_analysis: Optional[StackTraceAnalysis],
        artifact_analyses: List[ArtifactAnalysis],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform comprehensive failure analysis using AI model."""
        
        # Prepare analysis prompt
        analysis_prompt = self._build_analysis_prompt(
            execution_result,
            stack_trace_analysis,
            artifact_analyses,
            context
        )
        
        messages = [
            {
                "role": "system",
                "content": self._get_analysis_system_prompt()
            },
            {
                "role": "user",
                "content": analysis_prompt
            }
        ]
        
        try:
            # Route to OpenAI for complex analysis
            response = await self.model_router.route_task(
                TaskType.ANALYSIS,
                messages,
                context={"analysis_type": "failure_analysis"}
            )
            
            # Parse the AI response
            return self._parse_analysis_response(response.content)
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            # Fallback to rule-based analysis
            return self._fallback_analysis(execution_result, stack_trace_analysis)
    
    def _build_analysis_prompt(
        self,
        execution_result: ExecutionResult,
        stack_trace_analysis: Optional[StackTraceAnalysis],
        artifact_analyses: List[ArtifactAnalysis],
        context: Dict[str, Any]
    ) -> str:
        """Build the analysis prompt for the AI model."""
        
        prompt_parts = [
            f"# Test Failure Analysis Request",
            f"",
            f"## Test Information",
            f"- Test Name: {execution_result.test_name}",
            f"- Test File: {execution_result.test_file}",
            f"- Status: {execution_result.status.value}",
            f"- Duration: {execution_result.duration:.2f}s",
            f"- Exit Code: {execution_result.exit_code}",
            f"",
            f"## Error Information",
            f"- Error Message: {execution_result.error_message or 'Not available'}",
            f"- Error Type: {execution_result.error_type or 'Unknown'}",
            f"",
        ]
        
        # Add stack trace analysis
        if stack_trace_analysis:
            prompt_parts.extend([
                f"## Stack Trace Analysis",
                f"- Failing Line: {stack_trace_analysis.failing_line or 'Unknown'}",
                f"- Failing Code: {stack_trace_analysis.failing_code or 'Not available'}",
                f"- Playwright Action: {stack_trace_analysis.playwright_action or 'Unknown'}",
                f"- Selector Used: {stack_trace_analysis.selector_used or 'Not available'}",
                f"",
                f"### Full Stack Trace",
                f"```",
                stack_trace_analysis.raw_stack_trace[:2000],  # Limit size
                f"```",
                f"",
            ])
        
        # Add artifact analyses
        if artifact_analyses:
            prompt_parts.extend([
                f"## Artifact Analysis",
                f"",
            ])
            
            for artifact in artifact_analyses:
                prompt_parts.extend([
                    f"### {artifact.artifact_type.title()} Analysis",
                    f"- Findings: {', '.join(artifact.findings) if artifact.findings else 'None'}",
                    f"- Error Indicators: {', '.join(artifact.error_indicators) if artifact.error_indicators else 'None'}",
                    f"",
                ])
        
        # Add execution output
        if execution_result.stderr:
            prompt_parts.extend([
                f"## Standard Error Output",
                f"```",
                execution_result.stderr[:1000],  # Limit size
                f"```",
                f"",
            ])
        
        # Add context
        if context:
            prompt_parts.extend([
                f"## Additional Context",
                f"```json",
                json.dumps(context, indent=2)[:500],  # Limit size
                f"```",
                f"",
            ])
        
        prompt_parts.extend([
            f"## Analysis Request",
            f"Please analyze this test failure and provide:",
            f"1. Primary error category (selector, timing, network, assertion, navigation, element_state, timeout, browser_crash, other)",
            f"2. Root cause analysis",
            f"3. Fix suggestions with confidence levels",
            f"4. Overall confidence in the analysis",
            f"",
            f"Focus on actionable fixes that can be automatically applied to the test code.",
        ])
        
        return "\n".join(prompt_parts)
    
    def _get_analysis_system_prompt(self) -> str:
        """Get the system prompt for failure analysis."""
        return """You are an expert Playwright test failure analyzer. Your job is to analyze test failures and provide actionable fix suggestions.

When analyzing failures, consider:
1. Common Playwright issues (selectors, timing, element states)
2. Best practices for robust test automation
3. Specific error patterns and their typical solutions

Respond with a JSON object containing:
{
  "primary_error_category": "selector|timing|network|assertion|navigation|element_state|timeout|browser_crash|other",
  "secondary_categories": ["category1", "category2"],
  "root_cause": "Clear explanation of what caused the failure",
  "contributing_factors": ["factor1", "factor2"],
  "fix_suggestions": [
    {
      "fix_type": "selector_replacement|wait_condition|timeout_adjustment|assertion_update|navigation_fix|element_interaction|error_handling|retry_logic",
      "description": "Human-readable description",
      "confidence": "high|medium|low",
      "original_code": "code to replace (if applicable)",
      "suggested_code": "replacement code (if applicable)",
      "line_number": 42,
      "reasoning": "Why this fix should work",
      "potential_side_effects": ["effect1", "effect2"],
      "test_impact": "minimal|moderate|significant",
      "requires_manual_review": false
    }
  ],
  "overall_confidence": "high|medium|low",
  "similar_failures": ["pattern1", "pattern2"],
  "environment_factors": {"factor": "value"}
}

Prioritize fixes that:
- Use semantic selectors (getByRole, getByLabel, getByTestId)
- Add proper wait conditions instead of arbitrary timeouts
- Improve element interaction reliability
- Follow Playwright best practices"""
    
    def _parse_analysis_response(self, response_content: str) -> Dict[str, Any]:
        """Parse the AI analysis response into structured data."""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")
            
            analysis_data = json.loads(json_match.group())
            
            # Convert to our models
            result = {
                "primary_error_category": ErrorCategory(analysis_data["primary_error_category"]),
                "secondary_categories": [
                    ErrorCategory(cat) for cat in analysis_data.get("secondary_categories", [])
                ],
                "root_cause": analysis_data["root_cause"],
                "contributing_factors": analysis_data.get("contributing_factors", []),
                "overall_confidence": ConfidenceLevel(analysis_data["overall_confidence"]),
                "similar_failures": analysis_data.get("similar_failures", []),
                "environment_factors": analysis_data.get("environment_factors", {}),
                "fix_suggestions": []
            }
            
            # Parse fix suggestions
            for fix_data in analysis_data.get("fix_suggestions", []):
                fix_suggestion = FixSuggestion(
                    fix_type=FixType(fix_data["fix_type"]),
                    description=fix_data["description"],
                    confidence=ConfidenceLevel(fix_data["confidence"]),
                    original_code=fix_data.get("original_code"),
                    suggested_code=fix_data.get("suggested_code"),
                    line_number=fix_data.get("line_number"),
                    reasoning=fix_data["reasoning"],
                    potential_side_effects=fix_data.get("potential_side_effects", []),
                    test_impact=fix_data.get("test_impact", "minimal"),
                    requires_manual_review=fix_data.get("requires_manual_review", False)
                )
                result["fix_suggestions"].append(fix_suggestion)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse AI analysis response: {e}")
            raise FailureAnalysisError(f"Failed to parse analysis response: {e}")
    
    def _fallback_analysis(
        self,
        execution_result: ExecutionResult,
        stack_trace_analysis: Optional[StackTraceAnalysis]
    ) -> Dict[str, Any]:
        """Provide fallback rule-based analysis when AI analysis fails."""
        
        logger.info("Using fallback rule-based analysis")
        
        # Determine error category based on error message and type
        primary_category = ErrorCategory.OTHER
        root_cause = "Test failure detected"
        fix_suggestions = []
        
        if stack_trace_analysis:
            error_message = stack_trace_analysis.error_message.lower()
            
            if "timeout" in error_message:
                primary_category = ErrorCategory.TIMEOUT
                root_cause = "Test timed out waiting for an element or action"
                fix_suggestions.append(
                    FixSuggestion(
                        fix_type=FixType.TIMEOUT_ADJUSTMENT,
                        description="Increase timeout or add explicit wait condition",
                        confidence=ConfidenceLevel.MEDIUM,
                        reasoning="Timeout errors often need longer wait times or better wait conditions",
                        potential_side_effects=["Slower test execution"],
                        test_impact="minimal"
                    )
                )
            
            elif "selector" in error_message or "element" in error_message:
                primary_category = ErrorCategory.SELECTOR
                root_cause = "Element selector failed to find the target element"
                fix_suggestions.append(
                    FixSuggestion(
                        fix_type=FixType.SELECTOR_REPLACEMENT,
                        description="Update selector to be more robust",
                        confidence=ConfidenceLevel.MEDIUM,
                        reasoning="Selector failures often indicate brittle element targeting",
                        potential_side_effects=["May need to update multiple tests"],
                        test_impact="minimal"
                    )
                )
            
            elif "assertion" in error_message or "expect" in error_message:
                primary_category = ErrorCategory.ASSERTION
                root_cause = "Test assertion failed - expected condition not met"
                fix_suggestions.append(
                    FixSuggestion(
                        fix_type=FixType.ASSERTION_UPDATE,
                        description="Review and update test assertion",
                        confidence=ConfidenceLevel.LOW,
                        reasoning="Assertion failures may indicate changed application behavior",
                        potential_side_effects=["May mask real application issues"],
                        test_impact="significant",
                        requires_manual_review=True
                    )
                )
        
        return {
            "primary_error_category": primary_category,
            "secondary_categories": [],
            "root_cause": root_cause,
            "contributing_factors": ["Automated analysis unavailable"],
            "overall_confidence": ConfidenceLevel.LOW,
            "similar_failures": [],
            "environment_factors": {"analysis_method": "fallback_rules"},
            "fix_suggestions": fix_suggestions
        }