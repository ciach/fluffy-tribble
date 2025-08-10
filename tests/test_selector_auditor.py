"""
Unit tests for the selector auditor.
"""

import pytest
from unittest.mock import Mock, patch, mock_open

from orchestrator.generation.selector_auditor import SelectorAuditor
from orchestrator.generation.models import (
    SelectorViolation,
    AuditResult,
    SelectorType,
    ViolationSeverity,
)
from orchestrator.core.exceptions import ValidationError


class TestSelectorAuditor:
    """Test cases for SelectorAuditor."""

    @pytest.fixture
    def auditor(self):
        """Create a selector auditor with default policy."""
        with patch("pathlib.Path.exists", return_value=False):
            return SelectorAuditor()

    @pytest.fixture
    def sample_policy_content(self):
        """Sample policy file content."""
        return """
# Selector Policy

Prefer `getByRole`, `getByLabel`, or `getByTestId`.
Avoid brittle CSS unless justified.
"""

    def test_init_with_default_policy(self, auditor):
        """Test initialization with default policy when file doesn't exist."""
        assert auditor.policy_rules["preferred_selectors"] == [
            "getByRole",
            "getByLabel",
            "getByTestId",
        ]
        assert auditor.policy_rules["discouraged_selectors"] == ["css", "xpath"]
        assert auditor.policy_rules["allow_with_justification"] is True

    def test_init_with_policy_file(self, sample_policy_content):
        """Test initialization with policy file."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=sample_policy_content)):
                auditor = SelectorAuditor()
                assert "getByRole" in auditor.policy_rules["preferred_selectors"]

    def test_audit_compliant_code(self, auditor):
        """Test auditing compliant test code."""
        test_code = """
import { test, expect } from '@playwright/test';

test('login test', async ({ page }) => {
  await page.getByRole('button', { name: 'Login' }).click();
  await page.getByLabel('Username').fill('testuser');
  await page.getByTestId('password-input').fill('password');
  await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();
});
"""

        result = auditor.audit_test_code(test_code, "test.spec.ts")

        assert result.is_compliant is True
        assert len(result.violations) == 0
        assert result.total_selectors == 4
        assert result.compliant_selectors == 4
        assert result.compliance_rate == 100.0

    def test_audit_non_compliant_code(self, auditor):
        """Test auditing non-compliant test code."""
        test_code = """
import { test, expect } from '@playwright/test';

test('login test', async ({ page }) => {
  await page.locator('.login-button').click();
  await page.locator('#username').fill('testuser');
  await page.locator('//input[@type="password"]').fill('password');
});
"""

        result = auditor.audit_test_code(test_code, "test.spec.ts")

        assert result.is_compliant is False
        assert len(result.violations) >= 3  # May have more due to pattern matching
        assert result.compliant_selectors == 0

        # Check that we have both CSS and XPath violations
        css_violations = [
            v for v in result.violations if v.selector_type == SelectorType.CSS
        ]
        xpath_violations = [
            v for v in result.violations if v.selector_type == SelectorType.XPATH
        ]

        assert len(css_violations) >= 2
        assert len(xpath_violations) >= 1

    def test_audit_justified_violations(self, auditor):
        """Test auditing code with justified violations."""
        test_code = """
import { test, expect } from '@playwright/test';

test('login test', async ({ page }) => {
  // Legacy selector - justified for compatibility
  await page.locator('.legacy-button').click();
  await page.getByRole('button', { name: 'Submit' }).click();
});
"""

        result = auditor.audit_test_code(test_code, "test.spec.ts")

        assert (
            result.is_compliant is True
        )  # Justified violations don't block compliance
        assert len(result.violations) == 1
        assert result.violations[0].severity == ViolationSeverity.WARNING
        assert result.violations[0].justification == "Legacy selector - justified for compatibility"
        assert result.justified_violations == 1

    def test_audit_brittle_css_selectors(self, auditor):
        """Test detection of brittle CSS selectors."""
        test_code = """
test('brittle selectors', async ({ page }) => {
  await page.locator('div > div > div:nth-child(3)').click();
  await page.locator('.auto-generated-class-name-12345678901234567890').click();
  await page.locator('#auto-id-98765432109876543210').click();
});
"""

        result = auditor.audit_test_code(test_code, "test.spec.ts")

        # All CSS selectors will be discouraged, but some should also be brittle
        css_violations = [
            v for v in result.violations if v.selector_type == SelectorType.CSS
        ]
        assert len(css_violations) == 3  # All three should be detected as CSS

        # Check that at least some have brittle characteristics
        brittle_selectors = [
            v.selector
            for v in css_violations
            if auditor._is_brittle_css_selector(v.selector)
        ]
        assert (
            len(brittle_selectors) >= 2
        )  # Should detect nth-child and long class names

    def test_has_justification_comment(self, auditor):
        """Test justification comment detection."""
        assert auditor._has_justification_comment(
            "// This is justified for legacy support"
        )
        assert auditor._has_justification_comment(
            "/* Necessary for third-party integration */"
        )
        assert auditor._has_justification_comment(
            "await page.locator('.btn'); // Required by design"
        )
        assert auditor._has_justification_comment(
            "// Third-party library requirement"
        )
        assert auditor._has_justification_comment(
            "// No alternative available"
        )
        assert auditor._has_justification_comment(
            "/* Temporary workaround for API limitation */"
        )
        assert not auditor._has_justification_comment(
            "await page.locator('.btn'); // Regular comment"
        )

    def test_extract_justification_text(self, auditor):
        """Test justification text extraction."""
        text = auditor._extract_justification_text(
            "// This is justified for legacy support"
        )
        assert text == "This is justified for legacy support"

        text = auditor._extract_justification_text(
            "/* Necessary for third-party integration */"
        )
        assert text == "Necessary for third-party integration"

        text = auditor._extract_justification_text(
            "await page.locator('.btn'); // Regular comment"
        )
        assert text is None  # No justification keywords

    def test_get_selector_type(self, auditor):
        """Test selector type mapping."""
        assert auditor._get_selector_type("getByRole") == SelectorType.ROLE
        assert auditor._get_selector_type("getByLabel") == SelectorType.LABEL
        assert auditor._get_selector_type("getByTestId") == SelectorType.TEST_ID
        assert auditor._get_selector_type("css") == SelectorType.CSS
        assert auditor._get_selector_type("xpath") == SelectorType.XPATH
        assert auditor._get_selector_type("unknown") == SelectorType.OTHER

    def test_suggest_alternative(self, auditor):
        """Test alternative selector suggestions."""
        css_suggestion = auditor._suggest_alternative("css", ".button")
        assert "getByRole" in css_suggestion

        xpath_suggestion = auditor._suggest_alternative("xpath", "//button")
        assert "semantic selectors" in xpath_suggestion

    def test_is_brittle_css_selector(self, auditor):
        """Test brittle CSS selector detection."""
        assert auditor._is_brittle_css_selector("div:nth-child(3)")
        assert auditor._is_brittle_css_selector("div > div > div")
        assert auditor._is_brittle_css_selector(
            ".very-long-generated-class-name-12345678901234567890"
        )
        assert auditor._is_brittle_css_selector(
            "#very-long-generated-id-12345678901234567890"
        )
        assert not auditor._is_brittle_css_selector(".simple-class")
        assert not auditor._is_brittle_css_selector("#simple-id")

    def test_validate_compliance_success(self, auditor):
        """Test successful compliance validation."""
        audit_result = AuditResult(
            is_compliant=True, violations=[], total_selectors=5, compliant_selectors=5
        )

        assert auditor.validate_compliance(audit_result) is True

    def test_validate_compliance_with_warnings(self, auditor):
        """Test compliance validation with warnings (should pass)."""
        warning_violation = SelectorViolation(
            line_number=1,
            selector=".btn",
            selector_type=SelectorType.CSS,
            violation_type="discouraged_selector",
            message="Discouraged selector (justified)",
            severity=ViolationSeverity.WARNING,
        )

        audit_result = AuditResult(
            is_compliant=True,
            violations=[warning_violation],
            total_selectors=1,
            compliant_selectors=0,
            justified_violations=1,
        )

        assert auditor.validate_compliance(audit_result) is True

    def test_validate_compliance_failure(self, auditor):
        """Test compliance validation failure with errors."""
        error_violation = SelectorViolation(
            line_number=1,
            selector=".btn",
            selector_type=SelectorType.CSS,
            violation_type="discouraged_selector",
            message="Discouraged selector",
            severity=ViolationSeverity.ERROR,
        )

        audit_result = AuditResult(
            is_compliant=False,
            violations=[error_violation],
            total_selectors=1,
            compliant_selectors=0,
        )

        with pytest.raises(ValidationError, match="Selector policy violations found"):
            auditor.validate_compliance(audit_result)

    def test_selector_patterns_compilation(self, auditor):
        """Test that selector patterns are properly compiled."""
        patterns = auditor._selector_patterns

        assert "getByRole" in patterns
        assert "getByLabel" in patterns
        assert "getByTestId" in patterns
        assert "css" in patterns
        assert "xpath" in patterns

        # Test pattern matching
        role_match = patterns["getByRole"].search(
            "getByRole('button', { name: 'Submit' })"
        )
        assert role_match is not None

        css_match = patterns["css"].search("locator('.my-class')")
        assert css_match is not None

    def test_audit_result_compliance_rate(self):
        """Test compliance rate calculation."""
        result = AuditResult(
            is_compliant=False, total_selectors=10, compliant_selectors=7
        )

        assert result.compliance_rate == 70.0

        # Test zero selectors case
        empty_result = AuditResult(
            is_compliant=True, total_selectors=0, compliant_selectors=0
        )

        assert empty_result.compliance_rate == 100.0

    def test_generate_violation_report_compliant(self, auditor):
        """Test violation report generation for compliant code."""
        audit_result = AuditResult(
            is_compliant=True, 
            violations=[], 
            total_selectors=5, 
            compliant_selectors=5
        )

        report = auditor.generate_violation_report(audit_result, "test.spec.ts")
        
        assert "✅ Selector audit passed" in report
        assert "100.0% compliant" in report
        assert "5 selectors" in report

    def test_generate_violation_report_with_violations(self, auditor):
        """Test violation report generation with violations."""
        error_violation = SelectorViolation(
            line_number=5,
            selector=".btn",
            selector_type=SelectorType.CSS,
            violation_type="discouraged_selector",
            message="Discouraged selector type 'css' used",
            severity=ViolationSeverity.ERROR,
            suggested_fix="Use getByRole() instead"
        )

        warning_violation = SelectorViolation(
            line_number=10,
            selector=".legacy-btn",
            selector_type=SelectorType.CSS,
            violation_type="discouraged_selector",
            message="Discouraged selector type 'css' used (justified)",
            severity=ViolationSeverity.WARNING,
            justification="Legacy compatibility required"
        )

        audit_result = AuditResult(
            is_compliant=False,
            violations=[error_violation, warning_violation],
            total_selectors=3,
            compliant_selectors=1,
            justified_violations=1
        )

        report = auditor.generate_violation_report(audit_result, "test.spec.ts")
        
        assert "🔍 Selector Audit Report" in report
        assert "❌ ERRORS" in report
        assert "⚠️  WARNINGS" in report
        assert "Line 5:" in report
        assert "Line 10:" in report
        assert "Use getByRole() instead" in report
        assert "Legacy compatibility required" in report
        assert "📋 Policy Recommendations" in report

    def test_get_policy_summary(self, auditor):
        """Test policy summary generation."""
        summary = auditor.get_policy_summary()
        
        assert "📋 Current Selector Policy" in summary
        assert "✅ Preferred Selectors" in summary
        assert "❌ Discouraged Selectors" in summary
        assert "💬 Justification Policy" in summary
        assert "getByRole()" in summary
        assert "getByLabel()" in summary
        assert "getByTestId()" in summary

    def test_enhanced_policy_parsing(self, sample_policy_content):
        """Test enhanced policy file parsing."""
        enhanced_policy = """
# Selector Policy

## Preferred Selectors
Prefer getByRole, getByLabel, or getByTestId for semantic selection.
Use getByPlaceholder for form inputs when appropriate.

## Discouraged Selectors  
Avoid CSS selectors unless justified.
Don't use XPath selectors.
Discourage querySelector methods.

## Justification Rules
CSS selectors are allowed when justified with comments.
XPath selectors require justification for third-party compatibility.
"""
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=enhanced_policy)):
                auditor = SelectorAuditor()
                
                assert "getByRole" in auditor.policy_rules["preferred_selectors"]
                assert "getByPlaceholder" in auditor.policy_rules["preferred_selectors"]
                assert "css" in auditor.policy_rules["discouraged_selectors"]
                assert "xpath" in auditor.policy_rules["discouraged_selectors"]
                assert "querySelector" in auditor.policy_rules["discouraged_selectors"]
                assert auditor.policy_rules["allow_with_justification"] is True
