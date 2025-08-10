"""
Selector auditor for enforcing selector policies in generated tests.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .models import SelectorViolation, AuditResult, SelectorType, ViolationSeverity
from ..core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class SelectorAuditor:
    """Audits test code for selector policy compliance."""

    def __init__(self, policy_file_path: str = "policies/selector.md"):
        """Initialize the selector auditor.

        Args:
            policy_file_path: Path to the selector policy file
        """
        self.policy_file_path = policy_file_path
        self.policy_rules = self._load_policy_rules()
        self._selector_patterns = self._compile_selector_patterns()

    def _load_policy_rules(self) -> Dict[str, Any]:
        """Load selector policy rules from file."""
        try:
            policy_path = Path(self.policy_file_path)
            if not policy_path.exists():
                logger.warning(f"Policy file not found: {self.policy_file_path}")
                return self._get_default_policy()

            with open(policy_path, "r") as f:
                content = f.read()

            # Parse the simple policy format
            rules = {
                "preferred_selectors": ["getByRole", "getByLabel", "getByTestId"],
                "discouraged_selectors": ["css", "xpath"],
                "require_justification": ["css", "xpath"],
                "allow_with_justification": True,
            }

            # Extract any specific rules from the markdown content
            if "getByRole" in content:
                rules["preferred_selectors"].append("getByRole")
            if "getByLabel" in content:
                rules["preferred_selectors"].append("getByLabel")
            if "getByTestId" in content:
                rules["preferred_selectors"].append("getByTestId")
            if "CSS" in content or "css" in content:
                rules["discouraged_selectors"].append("css")

            return rules

        except Exception as e:
            logger.error(f"Failed to load policy rules: {e}")
            return self._get_default_policy()

    def _get_default_policy(self) -> Dict[str, Any]:
        """Get default policy rules."""
        return {
            "preferred_selectors": ["getByRole", "getByLabel", "getByTestId"],
            "discouraged_selectors": ["css", "xpath"],
            "require_justification": ["css", "xpath"],
            "allow_with_justification": True,
        }

    def _compile_selector_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for different selector types."""
        return {
            "getByRole": re.compile(r"getByRole\s*\(\s*['\"]([^'\"]+)['\"]"),
            "getByLabel": re.compile(r"getByLabel\s*\(\s*['\"]([^'\"]+)['\"]"),
            "getByTestId": re.compile(r"getByTestId\s*\(\s*['\"]([^'\"]+)['\"]"),
            "getByPlaceholder": re.compile(
                r"getByPlaceholder\s*\(\s*['\"]([^'\"]+)['\"]"
            ),
            "getByText": re.compile(r"getByText\s*\(\s*['\"]([^'\"]+)['\"]"),
            "css": re.compile(
                r"locator\s*\(\s*['\"]([^'\"//]+)['\"]"
            ),  # Exclude xpath patterns
            "xpath": re.compile(r"locator\s*\(\s*['\"]//([^'\"]*)['\"]"),
            "querySelector": re.compile(r"querySelector\s*\(\s*['\"]([^'\"]+)['\"]"),
            "querySelectorAll": re.compile(
                r"querySelectorAll\s*\(\s*['\"]([^'\"]+)['\"]"
            ),
        }

    def audit_test_code(self, test_code: str, file_path: str = "") -> AuditResult:
        """Audit test code for selector policy compliance.

        Args:
            test_code: The test code to audit
            file_path: Path to the test file (for context)

        Returns:
            AuditResult with compliance status and violations
        """
        logger.info(f"Auditing selector compliance for: {file_path}")

        violations = []
        total_selectors = 0
        compliant_selectors = 0
        justified_violations = 0

        lines = test_code.split("\n")

        for line_num, line in enumerate(lines, 1):
            # Check previous line for justification comments
            prev_line = lines[line_num - 2] if line_num > 1 else ""
            has_justification = self._has_justification_comment(
                line
            ) or self._has_justification_comment(prev_line)

            line_violations, line_selectors, line_compliant, line_justified = (
                self._audit_line(line, line_num, has_justification)
            )

            violations.extend(line_violations)
            total_selectors += line_selectors
            compliant_selectors += line_compliant
            justified_violations += line_justified

        # Determine overall compliance
        is_compliant = (
            len([v for v in violations if v.severity == ViolationSeverity.ERROR]) == 0
        )

        result = AuditResult(
            is_compliant=is_compliant,
            violations=violations,
            total_selectors=total_selectors,
            compliant_selectors=compliant_selectors,
            justified_violations=justified_violations,
        )

        logger.info(
            f"Audit completed: {result.compliance_rate:.1f}% compliant, "
            f"{len(violations)} violations"
        )

        return result

    def _audit_line(
        self, line: str, line_num: int, has_justification: bool = False
    ) -> tuple[List[SelectorViolation], int, int, int]:
        """Audit a single line of code.

        Args:
            line: The line of code to audit
            line_num: Line number in the file
            has_justification: Whether this line has justification

        Returns:
            Tuple of (violations, total_selectors, compliant_selectors, justified_violations)
        """
        violations = []
        total_selectors = 0
        compliant_selectors = 0
        justified_violations = 0

        # Check each selector pattern
        for selector_name, pattern in self._selector_patterns.items():
            matches = pattern.findall(line)

            for match in matches:
                total_selectors += 1
                selector_type = self._get_selector_type(selector_name)

                violation = self._check_selector_compliance(
                    selector_name, match, selector_type, line_num, has_justification
                )

                if violation:
                    violations.append(violation)
                    if has_justification:
                        justified_violations += 1
                else:
                    compliant_selectors += 1

        return violations, total_selectors, compliant_selectors, justified_violations

    def _has_justification_comment(self, line: str) -> bool:
        """Check if line has a justification comment."""
        # Look for comments that indicate justification
        comment_patterns = [
            r"//.*(?:justified|necessary|required|brittle|legacy)",
            r"/\*.*(?:justified|necessary|required|brittle|legacy).*\*/",
        ]

        for pattern in comment_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True

        return False

    def _get_selector_type(self, selector_name: str) -> SelectorType:
        """Map selector name to SelectorType enum."""
        mapping = {
            "getByRole": SelectorType.ROLE,
            "getByLabel": SelectorType.LABEL,
            "getByTestId": SelectorType.TEST_ID,
            "getByPlaceholder": SelectorType.PLACEHOLDER,
            "getByText": SelectorType.TEXT,
            "css": SelectorType.CSS,
            "xpath": SelectorType.XPATH,
            "querySelector": SelectorType.CSS,
            "querySelectorAll": SelectorType.CSS,
        }
        return mapping.get(selector_name, SelectorType.OTHER)

    def _check_selector_compliance(
        self,
        selector_name: str,
        selector_value: str,
        selector_type: SelectorType,
        line_num: int,
        has_justification: bool,
    ) -> Optional[SelectorViolation]:
        """Check if a selector complies with policy.

        Returns:
            SelectorViolation if non-compliant, None if compliant
        """
        # Check if selector is preferred
        if selector_name in self.policy_rules["preferred_selectors"]:
            return None  # Compliant

        # Check if selector is discouraged
        if selector_name in self.policy_rules["discouraged_selectors"]:
            severity = ViolationSeverity.ERROR
            message = f"Discouraged selector type '{selector_name}' used"
            suggested_fix = self._suggest_alternative(selector_name, selector_value)

            # If justification is provided and allowed, downgrade to warning
            if has_justification and self.policy_rules.get(
                "allow_with_justification", False
            ):
                severity = ViolationSeverity.WARNING
                message += " (justified)"

            return SelectorViolation(
                line_number=line_num,
                selector=selector_value,
                selector_type=selector_type,
                violation_type="discouraged_selector",
                message=message,
                severity=severity,
                suggested_fix=suggested_fix,
                justification="Justified by comment" if has_justification else None,
            )

        # Check for brittle CSS selectors (even if not explicitly discouraged)
        if selector_type == SelectorType.CSS and self._is_brittle_css_selector(
            selector_value
        ):
            return SelectorViolation(
                line_number=line_num,
                selector=selector_value,
                selector_type=selector_type,
                violation_type="brittle_selector",
                message="Potentially brittle CSS selector",
                severity=ViolationSeverity.WARNING,
                suggested_fix="Consider using getByRole, getByLabel, or getByTestId",
            )

        return None

    def _suggest_alternative(self, selector_name: str, selector_value: str) -> str:
        """Suggest alternative selector."""
        if selector_name in ["css", "querySelector", "querySelectorAll"]:
            return "Consider using getByRole(), getByLabel(), or getByTestId()"
        elif selector_name == "xpath":
            return "Consider using semantic selectors like getByRole() or getByLabel()"
        else:
            return "Use preferred selector methods"

    def _is_brittle_css_selector(self, selector: str) -> bool:
        """Check if CSS selector is potentially brittle."""
        brittle_patterns = [
            r"nth-child",
            r"nth-of-type",
            r">\s*div\s*>\s*div",  # Deep nesting
            r"\.[a-zA-Z0-9-_]{20,}",  # Very long class names (likely generated)
            r"#[a-zA-Z0-9-_]{20,}",  # Very long IDs (likely generated)
        ]

        for pattern in brittle_patterns:
            if re.search(pattern, selector):
                return True

        return False

    def validate_compliance(self, audit_result: AuditResult) -> bool:
        """Validate that audit result meets compliance requirements.

        Args:
            audit_result: Result from audit_test_code

        Returns:
            True if compliant, False otherwise

        Raises:
            ValidationError: If there are blocking violations
        """
        error_violations = [
            v for v in audit_result.violations if v.severity == ViolationSeverity.ERROR
        ]

        if error_violations:
            violation_details = [
                f"Line {v.line_number}: {v.message}" for v in error_violations
            ]

            raise ValidationError(
                message=f"Selector policy violations found: {len(error_violations)} errors",
                validation_type="selector_policy",
                violations=violation_details,
            )

        return audit_result.is_compliant
