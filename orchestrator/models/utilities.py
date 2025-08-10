"""
Model interaction utilities that integrate all components.

Provides high-level utilities for model interactions with templates,
context management, parsing, and rate limiting.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from ..core.config import Config
from ..core.exceptions import ModelError, ValidationError
from .router import ModelRouter
from .templates import PromptTemplateManager
from .parsers import ResponseParser
from .context import ContextManager, ContextStrategy
from .rate_limiter import ModelRateLimitManager
from .types import TaskType, ModelResponse, ModelProvider

logger = logging.getLogger(__name__)


class ModelInteractionManager:
    """
    High-level manager for model interactions.

    Integrates routing, templating, context management, parsing,
    and rate limiting for seamless model interactions.
    """

    def __init__(self, config: Config):
        """Initialize the model interaction manager."""
        self.config = config
        self.router = ModelRouter(config)
        self.template_manager = PromptTemplateManager()
        self.response_parser = ResponseParser()
        self.context_manager = ContextManager()
        self.rate_limit_manager = ModelRateLimitManager()

        logger.info("Model interaction manager initialized")

    async def execute_task(
        self,
        task_type: TaskType,
        context: Dict[str, Any],
        model_override: Optional[str] = None,
        context_strategy: ContextStrategy = ContextStrategy.PRIORITIZE,
    ) -> Dict[str, Any]:
        """
        Execute a complete model task with all utilities.

        Args:
            task_type: Type of task to execute
            context: Context data for the task
            model_override: Optional model name override
            context_strategy: Strategy for handling large context

        Returns:
            Dictionary containing:
            - parsed_response: Structured parsed response
            - raw_response: Original model response
            - metadata: Execution metadata

        Raises:
            ModelError: If model interaction fails
            ValidationError: If input validation fails
        """
        start_time = datetime.now()

        try:
            # 1. Prepare context
            prepared_context = self.context_manager.prepare_context(
                context=context,
                task_type=task_type,
                model_name=model_override,
                strategy=context_strategy,
            )

            # 2. Generate prompt from template
            messages = self.template_manager.format_prompt(task_type, prepared_context)

            # 3. Validate context size
            fits, total_tokens, max_tokens = self.context_manager.validate_context_size(
                messages, model_override
            )

            if not fits:
                logger.warning(
                    f"Context size ({total_tokens}) exceeds limit ({max_tokens})"
                )
                # Try more aggressive context reduction
                prepared_context = self.context_manager.prepare_context(
                    context=context,
                    task_type=task_type,
                    model_name=model_override,
                    strategy=ContextStrategy.TRUNCATE,
                )
                messages = self.template_manager.format_prompt(
                    task_type, prepared_context
                )

            # 4. Route task to appropriate model
            model_response = await self.router.route_task(
                task_type=task_type,
                messages=messages,
                context={"original_context_keys": list(context.keys())},
            )

            # 5. Parse response
            parsed_response = self.response_parser.parse_response(model_response)

            # 6. Prepare result
            execution_time = (datetime.now() - start_time).total_seconds()

            result = {
                "parsed_response": parsed_response,
                "raw_response": model_response,
                "metadata": {
                    "task_type": task_type.value,
                    "execution_time": execution_time,
                    "model_provider": model_response.provider.value,
                    "model_name": model_response.model_name,
                    "context_strategy": context_strategy.value,
                    "context_prepared": len(prepared_context),
                    "context_original": len(context),
                    "tokens_used": (
                        model_response.usage.get("total_tokens", 0)
                        if model_response.usage
                        else 0
                    ),
                    "success": parsed_response.is_valid,
                },
            }

            logger.info(
                f"Task {task_type.value} completed successfully in {execution_time:.2f}s "
                f"using {model_response.provider.value}"
            )

            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(
                f"Task {task_type.value} failed after {execution_time:.2f}s: {e}"
            )

            # Return error result
            return {
                "parsed_response": None,
                "raw_response": None,
                "metadata": {
                    "task_type": task_type.value,
                    "execution_time": execution_time,
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            }

    async def plan_tests(
        self, specification: str, existing_tests: str = "", **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a test planning task.

        Args:
            specification: Test specification to analyze
            existing_tests: Existing test coverage information
            **kwargs: Additional context

        Returns:
            Planning task result
        """
        context = {
            "specification": specification,
            "existing_tests": existing_tests,
            **kwargs,
        }

        return await self.execute_task(TaskType.PLANNING, context)

    async def debug_failure(
        self,
        test_name: str,
        error_message: str,
        stack_trace: str,
        test_code: str,
        artifacts: Optional[Dict[str, bool]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute a debugging task.

        Args:
            test_name: Name of the failing test
            error_message: Error message from test failure
            stack_trace: Stack trace from failure
            test_code: Code of the failing test
            artifacts: Available artifacts (screenshots, traces, etc.)
            **kwargs: Additional context

        Returns:
            Debugging task result
        """
        artifacts = artifacts or {}

        context = {
            "test_name": test_name,
            "error_message": error_message,
            "stack_trace": stack_trace,
            "test_code": test_code,
            "has_screenshots": artifacts.get("screenshots", False),
            "has_trace": artifacts.get("trace", False),
            "has_console_logs": artifacts.get("console_logs", False),
            "has_network_logs": artifacts.get("network_logs", False),
            "additional_context": kwargs.get("additional_context", ""),
            **kwargs,
        }

        return await self.execute_task(TaskType.DEBUGGING, context)

    async def draft_test(
        self,
        test_case_name: str,
        test_steps: List[str],
        expected_results: str,
        page_url: str,
        selectors: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute a test drafting task.

        Args:
            test_case_name: Name of the test case
            test_steps: List of test steps
            expected_results: Expected test results
            page_url: URL of the page to test
            selectors: Available selectors
            **kwargs: Additional requirements

        Returns:
            Drafting task result
        """
        context = {
            "test_case_name": test_case_name,
            "test_steps": "\n".join(
                f"{i+1}. {step}" for i, step in enumerate(test_steps)
            ),
            "expected_results": expected_results,
            "page_url": page_url,
            "selectors": selectors or {},
            "requirements": kwargs.get("requirements", ""),
            **kwargs,
        }

        return await self.execute_task(TaskType.DRAFTING, context)

    async def analyze_tests(
        self,
        analysis_type: str,
        test_files: List[str],
        test_results: Optional[str] = None,
        focus_areas: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute a test analysis task.

        Args:
            analysis_type: Type of analysis to perform
            test_files: List of test files to analyze
            test_results: Historical test results
            focus_areas: Specific areas to focus on
            **kwargs: Additional context

        Returns:
            Analysis task result
        """
        context = {
            "analysis_type": analysis_type,
            "test_files": "\n".join(test_files),
            "test_results": test_results or "",
            "focus_areas": focus_areas or [],
            "current_issues": kwargs.get("current_issues", ""),
            **kwargs,
        }

        return await self.execute_task(TaskType.ANALYSIS, context)

    async def generate_component(
        self,
        component_type: str,
        requirements: str,
        specifications: str,
        integration_context: str = "",
        constraints: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute a code generation task.

        Args:
            component_type: Type of component to generate
            requirements: Component requirements
            specifications: Detailed specifications
            integration_context: Context for integration
            constraints: List of constraints
            **kwargs: Additional context

        Returns:
            Generation task result
        """
        context = {
            "component_type": component_type,
            "requirements": requirements,
            "specifications": specifications,
            "integration_context": integration_context,
            "constraints": constraints or [],
            **kwargs,
        }

        return await self.execute_task(TaskType.GENERATION, context)

    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all model interaction components."""
        return {
            "router": self.router.get_routing_info(),
            "templates": {
                "available_templates": [
                    t.value for t in self.template_manager.list_available_templates()
                ]
            },
            "rate_limits": self.rate_limit_manager.get_usage_stats(),
            "context_manager": {
                "default_model": self.context_manager.model_name,
                "available_strategies": [s.value for s in ContextStrategy],
            },
        }

    def register_custom_template(self, template) -> None:
        """Register a custom prompt template."""
        self.template_manager.register_template(template)
        logger.info(f"Registered custom template for {template.task_type.value}")

    async def validate_setup(self) -> Dict[str, Any]:
        """Validate that all components are properly set up."""
        validation_results = {
            "router": {"status": "unknown", "details": {}},
            "templates": {"status": "unknown", "details": {}},
            "context_manager": {"status": "unknown", "details": {}},
            "rate_limiter": {"status": "unknown", "details": {}},
        }

        try:
            # Test router
            available_providers = self.router.get_available_providers()
            validation_results["router"] = {
                "status": "ok" if available_providers else "warning",
                "details": {
                    "available_providers": [p.value for p in available_providers],
                    "routing_rules": len(self.router.routing_rules),
                },
            }
        except Exception as e:
            validation_results["router"] = {
                "status": "error",
                "details": {"error": str(e)},
            }

        try:
            # Test templates
            templates = self.template_manager.list_available_templates()
            validation_results["templates"] = {
                "status": "ok",
                "details": {
                    "template_count": len(templates),
                    "available_types": [t.value for t in templates],
                },
            }
        except Exception as e:
            validation_results["templates"] = {
                "status": "error",
                "details": {"error": str(e)},
            }

        try:
            # Test context manager
            test_context = {"test": "small context"}
            prepared = self.context_manager.prepare_context(
                test_context, TaskType.PLANNING
            )
            validation_results["context_manager"] = {
                "status": "ok",
                "details": {
                    "test_preparation": "success",
                    "prepared_keys": list(prepared.keys()),
                },
            }
        except Exception as e:
            validation_results["context_manager"] = {
                "status": "error",
                "details": {"error": str(e)},
            }

        try:
            # Test rate limiter
            stats = self.rate_limit_manager.get_usage_stats()
            validation_results["rate_limiter"] = {
                "status": "ok",
                "details": {"providers": list(stats.keys()), "current_usage": stats},
            }
        except Exception as e:
            validation_results["rate_limiter"] = {
                "status": "error",
                "details": {"error": str(e)},
            }

        return validation_results
