"""
Comprehensive test fixtures for improving test coverage.

Tests common scenarios and edge cases across all components.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

from orchestrator.core.config import Config
from orchestrator.core.workflow import WorkflowContext, WorkflowManager
from orchestrator.planning.models import TestSpecification, TestPlan, TestCase
from orchestrator.execution.models import ExecutionResult, TestStatus
from orchestrator.analysis.models import FailureAnalysis, FixSuggestion
from orchestrator.mcp.connection_manager import MCPServerConfig
from orchestrator.reporting.models import TestRunReport, TestResultSummary


class TestComprehensiveFixtures:
    """Comprehensive test fixtures and scenarios."""

    @pytest.fixture
    def comprehensive_config(self):
        """Create comprehensive configuration for testing."""
        config = Config()
        config.openai_api_key = "test-key-12345"
        config.model_provider = "mixed"
        config.log_level = "DEBUG"
        config.ci_mode = True
        config.headless_mode = True
        config.artifact_retention_days = 30
        return config

    @pytest.fixture
    def mock_workflow_context(self, comprehensive_config):
        """Create mock workflow context."""
        return WorkflowContext(
            workflow_id="test-workflow-comprehensive",
            config=comprehensive_config,
            metadata={
                "test_suite": "comprehensive",
                "environment": "test",
                "user_id": "test-user-123",
            },
        )

    @pytest.fixture
    def complex_test_specification(self):
        """Create complex test specification."""
        return TestSpecification(
            id="complex-spec-001",
            name="E-commerce Checkout Flow",
            description="Complete e-commerce checkout process with multiple payment methods",
            requirements=[
                "User can add items to cart",
                "User can modify cart quantities",
                "User can remove items from cart",
                "User can proceed to checkout",
                "User can select shipping method",
                "User can enter billing information",
                "User can select payment method (credit card, PayPal, Apple Pay)",
                "User can complete purchase",
                "User receives confirmation email",
                "Order appears in user account history",
            ],
            priority="critical",
            tags=["e-commerce", "checkout", "payment", "integration"],
            estimated_duration=45,
            dependencies=["user-authentication", "product-catalog"],
            test_data_requirements=[
                "Valid user accounts",
                "Product inventory",
                "Payment method test data",
                "Shipping address data",
            ],
        )

    @pytest.fixture
    def comprehensive_test_plan(self, complex_test_specification):
        """Create comprehensive test plan."""
        test_cases = [
            TestCase(
                id="tc-001",
                name="Add Single Item to Cart",
                description="Verify user can add a single item to shopping cart",
                steps=[
                    "Navigate to product page",
                    "Select product options (size, color)",
                    "Click 'Add to Cart' button",
                    "Verify item appears in cart",
                ],
                expected_result="Item is added to cart with correct details",
                priority="high",
                tags=["cart", "add-item"],
            ),
            TestCase(
                id="tc-002",
                name="Modify Cart Quantities",
                description="Verify user can change item quantities in cart",
                steps=[
                    "Add item to cart",
                    "Navigate to cart page",
                    "Change quantity using +/- buttons",
                    "Verify total price updates",
                ],
                expected_result="Quantity and total price update correctly",
                priority="medium",
                tags=["cart", "quantity"],
            ),
            TestCase(
                id="tc-003",
                name="Complete Checkout Process",
                description="Verify full checkout flow with credit card payment",
                steps=[
                    "Add items to cart",
                    "Proceed to checkout",
                    "Enter shipping information",
                    "Select shipping method",
                    "Enter payment information",
                    "Review order",
                    "Complete purchase",
                ],
                expected_result="Order is successfully placed and confirmation is shown",
                priority="critical",
                tags=["checkout", "payment", "end-to-end"],
            ),
        ]

        return TestPlan(
            id="plan-comprehensive-001",
            specification_id=complex_test_specification.id,
            test_cases=test_cases,
            estimated_duration=30,
            setup_requirements=[
                "Clean test database",
                "Reset user session",
                "Clear browser cache",
            ],
            teardown_requirements=[
                "Clean up test orders",
                "Reset inventory",
                "Clear user cart",
            ],
        )

    @pytest.fixture
    def detailed_execution_result(self):
        """Create detailed execution result."""
        return ExecutionResult(
            success=False,
            test_results={"passed": 2, "failed": 1, "skipped": 0, "total": 3},
            artifacts={
                "trace_file": "checkout_flow_trace.zip",
                "screenshots": [
                    "cart_page.png",
                    "checkout_step1.png",
                    "payment_error.png",
                ],
                "console_logs": [
                    {
                        "level": "info",
                        "message": "Cart loaded successfully",
                        "timestamp": "2024-01-15T10:30:00Z",
                    },
                    {
                        "level": "error",
                        "message": "Payment validation failed: Invalid card number",
                        "timestamp": "2024-01-15T10:32:15Z",
                    },
                ],
                "network_logs": [
                    {
                        "url": "https://api.payment.com/validate",
                        "method": "POST",
                        "status": 400,
                        "response": {"error": "Invalid card number format"},
                    }
                ],
                "performance_metrics": {
                    "page_load_time": 2.3,
                    "api_response_time": 1.8,
                    "total_test_duration": 45.7,
                },
            },
            duration=45.7,
            test_file="e2e/checkout_flow.spec.ts",
            status=TestStatus.FAILED,
            exit_code=1,
            error_message="Payment validation test failed",
            browser_info={
                "name": "chromium",
                "version": "119.0.6045.105",
                "headless": True,
            },
        )

    @pytest.fixture
    def comprehensive_failure_analysis(self):
        """Create comprehensive failure analysis."""
        return FailureAnalysis(
            root_cause="Payment form validation error - card number format not accepted",
            error_category="validation",
            confidence=0.92,
            suggested_fixes=[
                FixSuggestion(
                    fix_type="selector_update",
                    description="Update card number input selector",
                    confidence="high",
                    original_code="await page.fill('#card-number', '4111111111111111')",
                    suggested_code="await page.fill('[data-testid=\"card-number-input\"]', '4111 1111 1111 1111')",
                    line_number=45,
                    reasoning="Use data-testid selector and format card number with spaces",
                    potential_side_effects=[
                        "May need to update other card number references"
                    ],
                    test_impact="minimal",
                ),
                FixSuggestion(
                    fix_type="wait_condition",
                    description="Add wait for payment form validation",
                    confidence="high",
                    original_code="await page.click('[data-testid=\"submit-payment\"]')",
                    suggested_code="await page.fill('[data-testid=\"card-number-input\"]', '4111 1111 1111 1111');\n  await expect(page.locator('[data-testid=\"card-error\"]')).toBeHidden();\n  await page.click('[data-testid=\"submit-payment\"]')",
                    line_number=47,
                    reasoning="Wait for validation to complete before submitting",
                    potential_side_effects=["Increased test execution time"],
                    test_impact="minimal",
                ),
            ],
            artifacts_analyzed=[
                "checkout_flow_trace.zip",
                "payment_error.png",
                "console.log",
            ],
            analysis_duration=3.2,
            stack_trace="Error: Payment validation failed\n  at PaymentForm.validate (payment.js:123)\n  at checkout.spec.ts:47",
            related_errors=["Card number format validation", "Payment gateway timeout"],
        )

    @pytest.fixture
    def mcp_server_configs(self):
        """Create comprehensive MCP server configurations."""
        return {
            "playwright": MCPServerConfig(
                name="playwright",
                command="uvx",
                args=["playwright-mcp-server", "--headless"],
                timeout=60,
                max_retries=3,
                disabled=False,
            ),
            "filesystem": MCPServerConfig(
                name="filesystem",
                command="uvx",
                args=["filesystem-mcp-server", "--sandbox", "e2e/"],
                timeout=30,
                max_retries=2,
                disabled=False,
            ),
            "git": MCPServerConfig(
                name="git",
                command="uvx",
                args=["git-mcp-server"],
                timeout=45,
                max_retries=3,
                disabled=False,
            ),
        }

    @pytest.fixture
    def comprehensive_test_report(self, detailed_execution_result):
        """Create comprehensive test report."""
        return TestRunReport(
            run_id="run-comprehensive-001",
            start_time=datetime.now() - timedelta(minutes=10),
            end_time=datetime.now(),
            duration=600.0,
            summary=TestResultSummary(
                total_tests=10, passed=7, failed=2, skipped=1, success_rate=0.7
            ),
            test_results=[detailed_execution_result],
            environment_info={
                "os": "linux",
                "browser": "chromium",
                "node_version": "18.17.0",
                "playwright_version": "1.40.0",
            },
            configuration={
                "headless": True,
                "timeout": 30000,
                "retries": 2,
                "workers": 4,
            },
            artifacts_summary={
                "total_size": 15728640,  # 15MB
                "file_count": 25,
                "trace_files": 3,
                "screenshots": 12,
                "videos": 2,
            },
        )

    def test_config_validation_comprehensive(self, comprehensive_config):
        """Test comprehensive configuration validation."""
        # Test valid configuration
        comprehensive_config.validate()

        # Test configuration serialization
        config_dict = comprehensive_config.to_dict()
        assert "openai_api_key" in config_dict
        assert config_dict["model_provider"] == "mixed"
        assert config_dict["ci_mode"] is True

    def test_workflow_context_comprehensive(self, mock_workflow_context):
        """Test comprehensive workflow context functionality."""
        # Test workflow context properties
        assert mock_workflow_context.workflow_id == "test-workflow-comprehensive"
        assert mock_workflow_context.metadata["test_suite"] == "comprehensive"

        # Test duration calculation
        duration = mock_workflow_context.duration
        assert duration >= 0

        # Test serialization
        context_dict = mock_workflow_context.to_dict()
        assert "workflow_id" in context_dict
        assert "metadata" in context_dict

    def test_test_specification_comprehensive(self, complex_test_specification):
        """Test comprehensive test specification."""
        # Test specification properties
        assert len(complex_test_specification.requirements) == 10
        assert complex_test_specification.priority == "critical"
        assert "e-commerce" in complex_test_specification.tags

        # Test estimated duration
        assert complex_test_specification.estimated_duration == 45

        # Test dependencies
        assert "user-authentication" in complex_test_specification.dependencies

    def test_test_plan_comprehensive(self, comprehensive_test_plan):
        """Test comprehensive test plan."""
        # Test test plan structure
        assert len(comprehensive_test_plan.test_cases) == 3
        assert comprehensive_test_plan.estimated_duration == 30

        # Test test cases
        critical_cases = [
            tc for tc in comprehensive_test_plan.test_cases if tc.priority == "critical"
        ]
        assert len(critical_cases) == 1

        # Test setup requirements
        assert "Clean test database" in comprehensive_test_plan.setup_requirements

    def test_execution_result_comprehensive(self, detailed_execution_result):
        """Test comprehensive execution result."""
        # Test result properties
        assert detailed_execution_result.success is False
        assert detailed_execution_result.test_results["total"] == 3
        assert detailed_execution_result.status == TestStatus.FAILED

        # Test artifacts
        assert "trace_file" in detailed_execution_result.artifacts
        assert len(detailed_execution_result.artifacts["screenshots"]) == 3
        assert len(detailed_execution_result.artifacts["console_logs"]) == 2

        # Test performance metrics
        assert "performance_metrics" in detailed_execution_result.artifacts
        assert (
            detailed_execution_result.artifacts["performance_metrics"][
                "total_test_duration"
            ]
            == 45.7
        )

    def test_failure_analysis_comprehensive(self, comprehensive_failure_analysis):
        """Test comprehensive failure analysis."""
        # Test analysis properties
        assert comprehensive_failure_analysis.confidence == 0.92
        assert comprehensive_failure_analysis.error_category == "validation"
        assert len(comprehensive_failure_analysis.suggested_fixes) == 2

        # Test fix suggestions
        selector_fixes = [
            fix
            for fix in comprehensive_failure_analysis.suggested_fixes
            if fix.fix_type == "selector_update"
        ]
        assert len(selector_fixes) == 1

        # Test analysis metadata
        assert comprehensive_failure_analysis.analysis_duration == 3.2
        assert len(comprehensive_failure_analysis.artifacts_analyzed) == 3

    def test_mcp_server_configs_comprehensive(self, mcp_server_configs):
        """Test comprehensive MCP server configurations."""
        # Test all required servers are present
        required_servers = ["playwright", "filesystem", "git"]
        for server in required_servers:
            assert server in mcp_server_configs
            assert mcp_server_configs[server].disabled is False

        # Test configuration properties
        playwright_config = mcp_server_configs["playwright"]
        assert playwright_config.timeout == 60
        assert playwright_config.max_retries == 3
        assert "--headless" in playwright_config.args

    def test_test_report_comprehensive(self, comprehensive_test_report):
        """Test comprehensive test report."""
        # Test report structure
        assert comprehensive_test_report.summary.total_tests == 10
        assert comprehensive_test_report.summary.success_rate == 0.7
        assert comprehensive_test_report.duration == 600.0

        # Test environment info
        assert comprehensive_test_report.environment_info["browser"] == "chromium"
        assert comprehensive_test_report.environment_info["os"] == "linux"

        # Test artifacts summary
        assert comprehensive_test_report.artifacts_summary["total_size"] == 15728640
        assert comprehensive_test_report.artifacts_summary["file_count"] == 25

    @pytest.mark.asyncio
    async def test_workflow_manager_comprehensive(self, comprehensive_config):
        """Test comprehensive workflow manager functionality."""
        manager = WorkflowManager(comprehensive_config)

        # Test workflow creation
        context = manager.start_workflow(metadata={"test": "comprehensive"})
        assert context is not None
        assert context.config == comprehensive_config

        # Test workflow tracking
        assert manager.current_workflow == context
        assert len(manager.active_workflows) == 1

    def test_edge_cases_and_error_conditions(self):
        """Test edge cases and error conditions."""
        # Test empty configuration
        empty_config = Config()
        with pytest.raises(Exception):
            empty_config.validate()

        # Test invalid test specification
        with pytest.raises(Exception):
            TestSpecification(
                id="",  # Empty ID should fail
                name="Test",
                description="Test description",
                requirements=[],
            )

        # Test invalid execution result
        with pytest.raises(Exception):
            ExecutionResult(
                success=True,
                test_results={"total": -1},  # Negative total should fail
                artifacts={},
                duration=-1.0,  # Negative duration should fail
                test_file="test.spec.ts",
                status=TestStatus.PASSED,
            )

    def test_performance_and_scalability_scenarios(self):
        """Test performance and scalability scenarios."""
        # Test large test specification
        large_requirements = [f"Requirement {i}" for i in range(100)]
        large_spec = TestSpecification(
            id="large-spec",
            name="Large Test Specification",
            description="Test with many requirements",
            requirements=large_requirements,
            priority="medium",
            tags=["performance", "scalability"],
        )

        assert len(large_spec.requirements) == 100

        # Test large test plan
        large_test_cases = []
        for i in range(50):
            test_case = TestCase(
                id=f"tc-{i:03d}",
                name=f"Test Case {i}",
                description=f"Test case number {i}",
                steps=[f"Step {j}" for j in range(5)],
                expected_result=f"Expected result {i}",
                priority="medium",
                tags=["generated", "performance"],
            )
            large_test_cases.append(test_case)

        large_plan = TestPlan(
            id="large-plan",
            specification_id="large-spec",
            test_cases=large_test_cases,
            estimated_duration=300,
        )

        assert len(large_plan.test_cases) == 50
        assert large_plan.estimated_duration == 300
