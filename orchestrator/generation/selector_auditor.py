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
        """Load selector policy rules from file with comprehensive parsing."""
        try:
            policy_path = Path(self.policy_file_path)
            if not policy_path.exists():
                logger.warning(f"Policy file not found: {self.policy_file_path}")
                return self._get_default_policy()

            with open(policy_path, "r") as f:
                content = f.read()

            # Start with default rules
            rules = self._get_default_policy()

            # Parse markdown content for policy rules
            lines = content.split("\n")
            current_section = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check for section headers
                if line.startswith("#"):
                    current_section = line.lower()
                    continue

                # Parse policy statements
                self._parse_policy_line(line, rules)

            # Log loaded policy for debugging
            logger.debug(f"Loaded policy rules: {rules}")
            return rules

        except Exception as e:
            logger.error(f"Failed to load policy rules: {e}")
            return self._get_default_policy()

    def _parse_policy_line(self, line: str, rules: Dict[str, Any]) -> None:
        """Parse a single policy line and update rules."""
        line_lower = line.lower()

        # Parse preferred selectors
        if any(keyword in line_lower for keyword in ["prefer", "use", "recommended"]):
            if "getbyrole" in line_lower:
                self._add_unique(rules["preferred_selectors"], "getByRole")
            if "getbylabel" in line_lower:
                self._add_unique(rules["preferred_selectors"], "getByLabel")
            if "getbytestid" in line_lower:
                self._add_unique(rules["preferred_selectors"], "getByTestId")
            if "getbyplaceholder" in line_lower:
                self._add_unique(rules["preferred_selectors"], "getByPlaceholder")
            if "getbytext" in line_lower:
                self._add_unique(rules["preferred_selectors"], "getByText")

        # Parse discouraged selectors
        if any(
            keyword in line_lower for keyword in ["avoid", "discourage", "don't use"]
        ):
            if "css" in line_lower:
                self._add_unique(rules["discouraged_selectors"], "css")
                self._add_unique(rules["require_justification"], "css")
            if "xpath" in line_lower:
                self._add_unique(rules["discouraged_selectors"], "xpath")
                self._add_unique(rules["require_justification"], "xpath")
            if "queryselector" in line_lower:
                self._add_unique(rules["discouraged_selectors"], "querySelector")
                self._add_unique(rules["require_justification"], "querySelector")

        # Parse justification requirements
        if any(keyword in line_lower for keyword in ["justify", "justified", "unless"]):
            rules["allow_with_justification"] = True
            if "css" in line_lower:
                self._add_unique(rules["require_justification"], "css")
            if "xpath" in line_lower:
                self._add_unique(rules["require_justification"], "xpath")

        # Parse strict enforcement
        if any(
            keyword in line_lower for keyword in ["never", "forbidden", "prohibited"]
        ):
            rules["allow_with_justification"] = False

    def _add_unique(self, list_obj: List[str], item: str) -> None:
        """Add item to list if not already present."""
        if item not in list_obj:
            list_obj.append(item)

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
            # Check current line and previous line for justification comments
            prev_line = lines[line_num - 2] if line_num > 1 else ""
            has_justification = self._has_justification_comment(
                line
            ) or self._has_justification_comment(prev_line)

            # Extract justification text from either line
            justification_text = self._extract_justification_text(
                line
            ) or self._extract_justification_text(prev_line)

            line_violations, line_selectors, line_compliant, line_justified = (
                self._audit_line(line, line_num, has_justification, justification_text)
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
        self,
        line: str,
        line_num: int,
        has_justification: bool = False,
        justification_text: Optional[str] = None,
    ) -> tuple[List[SelectorViolation], int, int, int]:
        """Audit a single line of code.

        Args:
            line: The line of code to audit
            line_num: Line number in the file
            has_justification: Whether this line has justification
            justification_text: The extracted justification text

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
                    selector_name,
                    match,
                    selector_type,
                    line_num,
                    has_justification,
                    justification_text,
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
            # Single line comments
            r"//.*(?:justified|necessary|required|brittle|legacy|third.?party|external|vendor|compatibility|design.?system|framework|library)",
            # Multi-line comments
            r"/\*.*(?:justified|necessary|required|brittle|legacy|third.?party|external|vendor|compatibility|design.?system|framework|library).*\*/",
            # Comments with specific justification phrases
            r"//.*(?:no alternative|only option|api limitation|browser limitation|temporary|workaround|migration)",
            r"/\*.*(?:no alternative|only option|api limitation|browser limitation|temporary|workaround|migration).*\*/",
        ]

        for pattern in comment_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True

        return False

    def _extract_justification_text(self, line: str) -> Optional[str]:
        """Extract the justification text from a comment."""
        # Extract comment content
        comment_patterns = [
            r"//\s*(.+)",  # Single line comment
            r"/\*\s*(.+?)\s*\*/",  # Multi-line comment
        ]

        for pattern in comment_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                comment_text = match.group(1).strip()
                # Check if it contains justification keywords
                if self._has_justification_comment(line):
                    return comment_text

        return None

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
        justification_text: Optional[str] = None,
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
                justification=justification_text if has_justification else None,
            )

        # Check for brittle CSS selectors (even if not explicitly discouraged)
        if selector_type == SelectorType.CSS and self._is_brittle_css_selector(
            selector_value
        ):
            severity = ViolationSeverity.WARNING
            message = "Potentially brittle CSS selector"

            # If justified, keep as warning but note justification
            if has_justification:
                message += " (justified)"

            return SelectorViolation(
                line_number=line_num,
                selector=selector_value,
                selector_type=selector_type,
                violation_type="brittle_selector",
                message=message,
                severity=severity,
                suggested_fix="Consider using getByRole, getByLabel, or getByTestId",
                justification=justification_text if has_justification else None,
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

    def generate_violation_report(
        self, audit_result: AuditResult, file_path: str = ""
    ) -> str:
        """Generate a comprehensive violation report.

        Args:
            audit_result: Result from audit_test_code
            file_path: Path to the audited file

        Returns:
            Formatted violation report
        """
        if audit_result.is_compliant and not audit_result.violations:
            return (
                f"✅ Selector audit passed for {file_path}\n"
                f"   {audit_result.total_selectors} selectors, {audit_result.compliance_rate:.1f}% compliant"
            )

        report_lines = []
        report_lines.append(f"🔍 Selector Audit Report for {file_path}")
        report_lines.append("=" * 60)

        # Summary
        report_lines.append(f"Total Selectors: {audit_result.total_selectors}")
        report_lines.append(f"Compliant: {audit_result.compliant_selectors}")
        report_lines.append(f"Violations: {len(audit_result.violations)}")
        report_lines.append(f"Justified: {audit_result.justified_violations}")
        report_lines.append(f"Compliance Rate: {audit_result.compliance_rate:.1f}%")
        report_lines.append("")

        # Group violations by severity
        errors = [
            v for v in audit_result.violations if v.severity == ViolationSeverity.ERROR
        ]
        warnings = [
            v
            for v in audit_result.violations
            if v.severity == ViolationSeverity.WARNING
        ]

        if errors:
            report_lines.append("❌ ERRORS (Must be fixed):")
            for violation in errors:
                report_lines.append(
                    f"   Line {violation.line_number}: {violation.message}"
                )
                report_lines.append(f"      Selector: '{violation.selector}'")
                if violation.suggested_fix:
                    report_lines.append(f"      Suggestion: {violation.suggested_fix}")
                if violation.justification:
                    report_lines.append(
                        f"      Justification: {violation.justification}"
                    )
                report_lines.append("")

        if warnings:
            report_lines.append("⚠️  WARNINGS:")
            for violation in warnings:
                report_lines.append(
                    f"   Line {violation.line_number}: {violation.message}"
                )
                report_lines.append(f"      Selector: '{violation.selector}'")
                if violation.suggested_fix:
                    report_lines.append(f"      Suggestion: {violation.suggested_fix}")
                if violation.justification:
                    report_lines.append(
                        f"      Justification: {violation.justification}"
                    )
                report_lines.append("")

        # Policy recommendations
        if errors or warnings:
            report_lines.append("📋 Policy Recommendations:")
            report_lines.append("   • Prefer getByRole(), getByLabel(), getByTestId()")
            report_lines.append(
                "   • Avoid CSS selectors unless justified with comments"
            )
            report_lines.append("   • Use semantic selectors for better test stability")
            report_lines.append(
                "   • Add justification comments for necessary brittle selectors"
            )

        return "\n".join(report_lines)

    def get_policy_summary(self) -> str:
        """Get a summary of the current policy rules.

        Returns:
            Formatted policy summary
        """
        summary_lines = []
        summary_lines.append("📋 Current Selector Policy:")
        summary_lines.append("=" * 40)

        summary_lines.append("✅ Preferred Selectors:")
        for selector in self.policy_rules["preferred_selectors"]:
            summary_lines.append(f"   • {selector}()")

        summary_lines.append("\n❌ Discouraged Selectors:")
        for selector in self.policy_rules["discouraged_selectors"]:
            summary_lines.append(f"   • {selector}")

        if self.policy_rules.get("allow_with_justification", False):
            summary_lines.append("\n💬 Justification Policy:")
            summary_lines.append(
                "   • Discouraged selectors allowed with justification comments"
            )
            summary_lines.append(
                "   • Use comments containing: justified, necessary, required, etc."
            )

        return "\n".join(summary_lines)
