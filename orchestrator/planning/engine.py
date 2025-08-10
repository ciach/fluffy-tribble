"""
Planning engine for analyzing test specifications and generating test plans.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..models.router import ModelRouter
from ..models.types import TaskType
from ..core.exceptions import PlanningError
from .models import (
    TestSpecification,
    TestPlan,
    TestCase,
    TestStep,
    Assertion,
    PageObject,
    SelectorInfo,
    TestGapAnalysis,
    Priority,
)

logger = logging.getLogger(__name__)


class PlanningEngine:
    """Engine for test specification analysis and test plan generation."""

    def __init__(self, model_router: ModelRouter):
        """Initialize the planning engine.

        Args:
            model_router: Router for AI model interactions
        """
        self.model_router = model_router
        self._planning_prompts = self._load_planning_prompts()

    def _load_planning_prompts(self) -> Dict[str, str]:
        """Load planning prompt templates."""
        return {
            "analyze_specification": """
Analyze the following test specification and extract key information:

Specification:
{specification}

Please provide a structured analysis including:
1. Core functionality to test
2. User workflows to cover
3. Edge cases to consider
4. Required test data/setup
5. Priority assessment

Respond in JSON format with the following structure:
{{
    "core_functionality": ["list", "of", "functions"],
    "user_workflows": ["list", "of", "workflows"],
    "edge_cases": ["list", "of", "edge", "cases"],
    "setup_requirements": ["list", "of", "setup", "needs"],
    "priority_assessment": "high|medium|low",
    "estimated_complexity": "simple|moderate|complex"
}}
""",
            "generate_test_plan": """
Based on the specification analysis, generate a comprehensive test plan:

Analysis:
{analysis}

Original Specification:
{specification}

Create a detailed test plan with the following structure:
1. Test cases covering all workflows
2. Page objects for reusable components
3. Setup requirements
4. Estimated durations

Respond in JSON format with this structure:
{{
    "test_cases": [
        {{
            "name": "test_case_name",
            "description": "what this test validates",
            "steps": [
                {{
                    "action": "navigate|click|fill|wait|assert",
                    "target": "element description",
                    "value": "value if applicable",
                    "description": "step description"
                }}
            ],
            "assertions": [
                {{
                    "type": "visibility|text|attribute|url",
                    "target": "element or condition to check",
                    "expected": "expected value",
                    "description": "what this assertion validates"
                }}
            ],
            "setup_requirements": ["list", "of", "setup", "needs"],
            "estimated_duration": 30.0
        }}
    ],
    "page_objects": [
        {{
            "name": "PageName",
            "url_pattern": "/path/pattern",
            "selectors": [
                {{
                    "selector": "getByRole('button', {{ name: 'Submit' }})",
                    "type": "role",
                    "element_description": "submit button",
                    "is_compliant": true
                }}
            ],
            "methods": ["login", "fillForm", "submit"]
        }}
    ],
    "setup_requirements": ["global", "setup", "needs"],
    "estimated_duration": 120.0
}}
""",
            "analyze_test_gaps": """
Analyze the existing test files and identify gaps compared to the specification:

Specification Requirements:
{specification}

Existing Test Files:
{existing_tests}

Identify:
1. Missing test cases
2. Incomplete coverage areas
3. Suggested additions
4. Priority gaps that need immediate attention

Respond in JSON format:
{{
    "missing_test_cases": ["list", "of", "missing", "tests"],
    "incomplete_coverage": ["areas", "with", "partial", "coverage"],
    "suggested_additions": ["specific", "test", "suggestions"],
    "priority_gaps": ["high", "priority", "missing", "tests"]
}}
""",
        }

    def analyze_specification(self, specification: TestSpecification) -> Dict[str, Any]:
        """Analyze a test specification to understand requirements.

        Args:
            specification: The test specification to analyze

        Returns:
            Dictionary containing analysis results

        Raises:
            PlanningError: If analysis fails
        """
        try:
            logger.info(f"Analyzing specification: {specification.name}")

            prompt = self._planning_prompts["analyze_specification"].format(
                specification=json.dumps(specification.to_dict(), indent=2)
            )

            response = self.model_router.generate_response(
                prompt=prompt, task_type=TaskType.PLANNING
            )

            # Parse JSON response
            analysis = json.loads(response.content)

            logger.info(f"Specification analysis completed for: {specification.name}")
            return analysis

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse analysis response: {e}")
            raise PlanningError(f"Invalid analysis response format: {e}")
        except Exception as e:
            logger.error(f"Specification analysis failed: {e}")
            raise PlanningError(f"Analysis failed: {e}")

    def create_test_plan(self, specification: TestSpecification) -> TestPlan:
        """Generate a comprehensive test plan from specification.

        Args:
            specification: The test specification

        Returns:
            Complete test plan

        Raises:
            PlanningError: If plan generation fails
        """
        try:
            logger.info(f"Creating test plan for: {specification.name}")

            # First analyze the specification
            analysis = self.analyze_specification(specification)

            # Generate the test plan
            prompt = self._planning_prompts["generate_test_plan"].format(
                analysis=json.dumps(analysis, indent=2),
                specification=json.dumps(specification.to_dict(), indent=2),
            )

            response = self.model_router.generate_response(
                prompt=prompt, task_type=TaskType.PLANNING
            )

            # Parse and convert to test plan
            plan_data = json.loads(response.content)
            test_plan = self._convert_to_test_plan(specification.id, plan_data)

            logger.info(
                f"Test plan created with {len(test_plan.test_cases)} test cases"
            )
            return test_plan

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse test plan response: {e}")
            raise PlanningError(f"Invalid test plan response format: {e}")
        except Exception as e:
            logger.error(f"Test plan creation failed: {e}")
            raise PlanningError(f"Plan creation failed: {e}")

    def identify_test_gaps(
        self, specification: TestSpecification, existing_tests: List[str]
    ) -> TestGapAnalysis:
        """Analyze gaps between specification and existing tests.

        Args:
            specification: The test specification
            existing_tests: List of existing test file contents

        Returns:
            Gap analysis results

        Raises:
            PlanningError: If gap analysis fails
        """
        try:
            logger.info(f"Analyzing test gaps for: {specification.name}")

            prompt = self._planning_prompts["analyze_test_gaps"].format(
                specification=json.dumps(specification.to_dict(), indent=2),
                existing_tests="\n\n".join(existing_tests),
            )

            response = self.model_router.generate_response(
                prompt=prompt, task_type=TaskType.ANALYSIS
            )

            # Parse and convert to gap analysis
            gap_data = json.loads(response.content)
            gap_analysis = TestGapAnalysis(
                missing_test_cases=gap_data.get("missing_test_cases", []),
                incomplete_coverage=gap_data.get("incomplete_coverage", []),
                suggested_additions=gap_data.get("suggested_additions", []),
                priority_gaps=gap_data.get("priority_gaps", []),
            )

            logger.info(
                f"Gap analysis completed: {len(gap_analysis.missing_test_cases)} missing tests"
            )
            return gap_analysis

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse gap analysis response: {e}")
            raise PlanningError(f"Invalid gap analysis response format: {e}")
        except Exception as e:
            logger.error(f"Gap analysis failed: {e}")
            raise PlanningError(f"Gap analysis failed: {e}")

    def _convert_to_test_plan(
        self, specification_id: str, plan_data: Dict[str, Any]
    ) -> TestPlan:
        """Convert JSON plan data to TestPlan object.

        Args:
            specification_id: ID of the source specification
            plan_data: JSON data from model response

        Returns:
            TestPlan object
        """
        # Convert test cases
        test_cases = []
        for tc_data in plan_data.get("test_cases", []):
            steps = [
                TestStep(
                    action=step["action"],
                    target=step["target"],
                    value=step.get("value"),
                    description=step.get("description", ""),
                )
                for step in tc_data.get("steps", [])
            ]

            assertions = [
                Assertion(
                    type=assertion["type"],
                    target=assertion["target"],
                    expected=assertion["expected"],
                    description=assertion.get("description", ""),
                )
                for assertion in tc_data.get("assertions", [])
            ]

            test_case = TestCase(
                name=tc_data["name"],
                description=tc_data["description"],
                steps=steps,
                assertions=assertions,
                setup_requirements=tc_data.get("setup_requirements", []),
                estimated_duration=tc_data.get("estimated_duration", 0.0),
            )
            test_cases.append(test_case)

        # Convert page objects
        page_objects = []
        for po_data in plan_data.get("page_objects", []):
            selectors = [
                SelectorInfo(
                    selector=sel["selector"],
                    type=sel["type"],
                    element_description=sel["element_description"],
                    is_compliant=sel.get("is_compliant", True),
                    justification=sel.get("justification"),
                )
                for sel in po_data.get("selectors", [])
            ]

            page_object = PageObject(
                name=po_data["name"],
                url_pattern=po_data["url_pattern"],
                selectors=selectors,
                methods=po_data.get("methods", []),
            )
            page_objects.append(page_object)

        return TestPlan(
            specification_id=specification_id,
            test_cases=test_cases,
            page_objects=page_objects,
            setup_requirements=plan_data.get("setup_requirements", []),
            estimated_duration=plan_data.get("estimated_duration", 0.0),
        )
