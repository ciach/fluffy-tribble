"""
Test generator for creating Playwright TypeScript test files.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..models.router import ModelRouter
from ..models.types import TaskType
from ..planning.models import TestPlan, TestCase, PageObject
from ..core.exceptions import ValidationError
from .models import GeneratedTest, PageObjectScaffold
from .selector_auditor import SelectorAuditor

logger = logging.getLogger(__name__)


class TestGenerator:
    """Generates Playwright TypeScript test files from test plans."""

    def __init__(
        self,
        model_router: ModelRouter,
        selector_auditor: Optional[SelectorAuditor] = None,
    ):
        """Initialize the test generator.

        Args:
            model_router: Router for AI model interactions
            selector_auditor: Auditor for selector policy enforcement
        """
        self.model_router = model_router
        self.selector_auditor = selector_auditor or SelectorAuditor()
        self._generation_prompts = self._load_generation_prompts()

    def _load_generation_prompts(self) -> Dict[str, str]:
        """Load test generation prompt templates."""
        return {
            "generate_test": """
Generate a Playwright TypeScript test file based on the following test plan:

Test Plan:
{test_plan}

Requirements:
1. Use TypeScript syntax with proper typing
2. Follow Playwright best practices
3. Use semantic selectors (getByRole, getByLabel, getByTestId) whenever possible
4. Avoid brittle CSS selectors unless absolutely necessary
5. Include proper test setup and teardown
6. Add meaningful test descriptions and comments
7. Use expect() assertions with proper matchers
8. Include proper error handling

Generate a complete test file with:
- Proper imports
- Test setup/teardown if needed
- Individual test cases for each scenario
- Page object usage where appropriate
- Proper async/await handling

File name: {file_name}

Respond with the complete TypeScript test file content:
""",
            "generate_page_object": """
Generate a Playwright TypeScript page object class based on the following specification:

Page Object Specification:
{page_object_spec}

Requirements:
1. Use TypeScript class syntax with proper typing
2. Include constructor that takes a Page object
3. Define locators using semantic selectors (getByRole, getByLabel, getByTestId)
4. Create methods for common page interactions
5. Include proper error handling and waits
6. Add JSDoc comments for methods
7. Follow Playwright page object patterns

Generate a complete page object class file:
""",
            "update_test": """
Update the existing Playwright test file with the following changes:

Current Test File:
{current_test}

Requested Changes:
{changes}

Requirements:
1. Maintain existing test structure where possible
2. Use semantic selectors for any new elements
3. Preserve existing working functionality
4. Add proper TypeScript typing for new code
5. Follow Playwright best practices
6. Ensure all changes are properly integrated

Respond with the complete updated test file content:
""",
        }

    def generate_test(
        self, test_plan: TestPlan, file_name: str, output_dir: str = "e2e"
    ) -> GeneratedTest:
        """Generate a complete test file from a test plan.

        Args:
            test_plan: The test plan to generate from
            file_name: Name for the generated test file
            output_dir: Directory to place the test file

        Returns:
            GeneratedTest with content and audit results

        Raises:
            ValidationError: If generated test fails selector audit
        """
        logger.info(f"Generating test file: {file_name}")

        try:
            # Generate the test content
            prompt = self._generation_prompts["generate_test"].format(
                test_plan=self._format_test_plan(test_plan), file_name=file_name
            )

            response = self.model_router.generate_response(
                prompt=prompt,
                task_type=TaskType.DRAFTING,  # Use local model for initial draft
            )

            test_content = response.content
            file_path = f"{output_dir}/{file_name}"

            # Create generated test object
            generated_test = GeneratedTest(
                name=file_name,
                content=test_content,
                file_path=file_path,
                test_plan_id=test_plan.specification_id,
                page_objects=self._extract_page_objects(test_plan),
                imports=self._extract_imports(test_content),
            )

            # Audit the generated test
            audit_result = self.selector_auditor.audit_test_code(
                test_content, file_path
            )
            generated_test.audit_result = audit_result

            # Generate comprehensive violation report
            violation_report = self.selector_auditor.generate_violation_report(
                audit_result, file_path
            )
            logger.info(f"Selector audit report:\n{violation_report}")

            # Validate compliance (this will raise ValidationError if non-compliant)
            try:
                self.selector_auditor.validate_compliance(audit_result)
                logger.info(
                    f"Test generation completed: {file_name} "
                    f"({audit_result.compliance_rate:.1f}% compliant)"
                )
            except ValidationError as e:
                logger.error(f"Selector policy violations prevent test generation: {e}")
                logger.error(f"Violation details:\n{violation_report}")
                raise

            return generated_test

        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error(f"Test generation failed: {e}")
            raise ValidationError(f"Test generation failed: {e}")

    def update_test(
        self, existing_test: str, changes: List[str], file_path: str
    ) -> GeneratedTest:
        """Update an existing test file with changes.

        Args:
            existing_test: Current test file content
            changes: List of changes to apply
            file_path: Path to the test file

        Returns:
            GeneratedTest with updated content

        Raises:
            ValidationError: If updated test fails selector audit
        """
        logger.info(f"Updating test file: {file_path}")

        try:
            prompt = self._generation_prompts["update_test"].format(
                current_test=existing_test, changes="\n".join(changes)
            )

            response = self.model_router.generate_response(
                prompt=prompt, task_type=TaskType.GENERATION
            )

            updated_content = response.content

            # Create updated test object
            updated_test = GeneratedTest(
                name=Path(file_path).name,
                content=updated_content,
                file_path=file_path,
                test_plan_id="updated",
                imports=self._extract_imports(updated_content),
            )

            # Audit the updated test
            audit_result = self.selector_auditor.audit_test_code(
                updated_content, file_path
            )
            updated_test.audit_result = audit_result

            # Generate comprehensive violation report
            violation_report = self.selector_auditor.generate_violation_report(
                audit_result, file_path
            )
            logger.info(f"Selector audit report:\n{violation_report}")

            # Validate compliance
            try:
                self.selector_auditor.validate_compliance(audit_result)
                logger.info(f"Test update completed: {file_path}")
            except ValidationError as e:
                logger.error(f"Selector policy violations prevent test update: {e}")
                logger.error(f"Violation details:\n{violation_report}")
                raise

            return updated_test

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Test update failed: {e}")
            raise ValidationError(f"Test update failed: {e}")

    def scaffold_page_object(
        self, page_object: PageObject, output_dir: str = "e2e/pages"
    ) -> PageObjectScaffold:
        """Generate a page object class file.

        Args:
            page_object: Page object specification
            output_dir: Directory for page object files

        Returns:
            PageObjectScaffold with generated content
        """
        logger.info(f"Scaffolding page object: {page_object.name}")

        try:
            prompt = self._generation_prompts["generate_page_object"].format(
                page_object_spec=self._format_page_object(page_object)
            )

            response = self.model_router.generate_response(
                prompt=prompt, task_type=TaskType.DRAFTING
            )

            class_name = page_object.name
            file_name = f"{class_name.lower().replace('page', '')}.page.ts"
            file_path = f"{output_dir}/{file_name}"

            scaffold = PageObjectScaffold(
                class_name=class_name,
                file_path=file_path,
                url_pattern=page_object.url_pattern,
                selectors={
                    sel.element_description: sel.selector
                    for sel in page_object.selectors
                },
                methods=page_object.methods,
                imports=["import { Page, Locator } from '@playwright/test';"],
            )

            logger.info(f"Page object scaffolded: {class_name}")
            return scaffold

        except Exception as e:
            logger.error(f"Page object scaffolding failed: {e}")
            raise ValidationError(f"Page object scaffolding failed: {e}")

    def _format_test_plan(self, test_plan: TestPlan) -> str:
        """Format test plan for prompt inclusion."""
        formatted = f"Specification ID: {test_plan.specification_id}\n"
        formatted += f"Estimated Duration: {test_plan.estimated_duration}s\n\n"

        formatted += "Test Cases:\n"
        for i, test_case in enumerate(test_plan.test_cases, 1):
            formatted += f"{i}. {test_case.name}\n"
            formatted += f"   Description: {test_case.description}\n"
            formatted += f"   Steps:\n"
            for j, step in enumerate(test_case.steps, 1):
                formatted += f"      {j}. {step.action}: {step.target}"
                if step.value:
                    formatted += f" = '{step.value}'"
                formatted += f" ({step.description})\n"

            formatted += f"   Assertions:\n"
            for j, assertion in enumerate(test_case.assertions, 1):
                formatted += f"      {j}. {assertion.type}: {assertion.target} should be '{assertion.expected}'\n"
            formatted += "\n"

        if test_plan.page_objects:
            formatted += "Page Objects:\n"
            for page_obj in test_plan.page_objects:
                formatted += f"- {page_obj.name} ({page_obj.url_pattern})\n"

        return formatted

    def _format_page_object(self, page_object: PageObject) -> str:
        """Format page object for prompt inclusion."""
        formatted = f"Class Name: {page_object.name}\n"
        formatted += f"URL Pattern: {page_object.url_pattern}\n\n"

        formatted += "Selectors:\n"
        for selector in page_object.selectors:
            formatted += f"- {selector.element_description}: {selector.selector}\n"

        if page_object.methods:
            formatted += f"\nMethods: {', '.join(page_object.methods)}\n"

        return formatted

    def _extract_page_objects(self, test_plan: TestPlan) -> List[str]:
        """Extract page object names from test plan."""
        return [po.name for po in test_plan.page_objects]

    def _extract_imports(self, test_content: str) -> List[str]:
        """Extract import statements from test content."""
        imports = []
        lines = test_content.split("\n")

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("import ") and stripped.endswith(";"):
                imports.append(stripped)

        return imports
