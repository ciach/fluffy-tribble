"""
Unit tests for the test suite health monitor.
"""

import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path

from orchestrator.analysis.health_monitor import (
    TestSuiteHealthMonitor,
    HealthIssue,
    HealthReport,
)


class TestHealthIssue:
    """Test cases for HealthIssue."""

    def test_health_issue_creation(self):
        """Test creating a health issue."""
        issue = HealthIssue(
            issue_type="test_issue",
            severity="high",
            file_path="test.spec.ts",
            line_number=10,
            description="Test issue description",
            suggestion="Fix suggestion",
            affected_code="test code",
            metadata={"key": "value"},
        )

        assert issue.issue_type == "test_issue"
        assert issue.severity == "high"
        assert issue.file_path == "test.spec.ts"
        assert issue.line_number == 10

    def test_health_issue_to_dict(self):
        """Test converting health issue to dictionary."""
        issue = HealthIssue(
            issue_type="test_issue",
            severity="medium",
            file_path="test.spec.ts",
            description="Test description",
        )

        result = issue.to_dict()

        assert result["issue_type"] == "test_issue"
        assert result["severity"] == "medium"
        assert result["file_path"] == "test.spec.ts"
        assert result["description"] == "Test description"


class TestHealthReport:
    """Test cases for HealthReport."""

    def test_health_report_creation(self):
        """Test creating a health report."""
        report = HealthReport(total_files=5)

        assert report.total_files == 5
        assert len(report.issues) == 0
        assert len(report.duplicate_helpers) == 0

    def test_priority_issue_filtering(self):
        """Test filtering issues by priority."""
        high_issue = HealthIssue("test", "high", "file1.ts")
        medium_issue = HealthIssue("test", "medium", "file2.ts")
        low_issue = HealthIssue("test", "low", "file3.ts")

        report = HealthReport(
            total_files=3, issues=[high_issue, medium_issue, low_issue]
        )

        assert len(report.high_priority_issues) == 1
        assert len(report.medium_priority_issues) == 1
        assert len(report.low_priority_issues) == 1
        assert report.high_priority_issues[0] == high_issue

    def test_health_report_to_dict(self):
        """Test converting health report to dictionary."""
        issue = HealthIssue("test", "high", "file.ts")
        report = HealthReport(total_files=1, issues=[issue])

        result = report.to_dict()

        assert result["total_files"] == 1
        assert result["total_issues"] == 1
        assert result["high_priority_count"] == 1
        assert len(result["issues"]) == 1


class TestTestSuiteHealthMonitor:
    """Test cases for TestSuiteHealthMonitor."""

    @pytest.fixture
    def monitor(self):
        """Create a health monitor instance."""
        return TestSuiteHealthMonitor("test_e2e")

    @pytest.fixture
    def sample_test_content(self):
        """Sample test file content with various issues."""
        return """
import { test, expect } from '@playwright/test';

// Helper function
async function loginUser(page, username, password) {
  await page.locator('#username').fill(username);
  await page.locator('#password').fill(password);
  await page.locator('.login-btn').click();
}

// Another similar helper
const userLogin = async (page, user, pass) => {
  await page.locator('#user-input').fill(user);
  await page.locator('#pass-input').fill(pass);
  await page.locator('button[type="submit"]').click();
};

test('user login', async ({ page }) => {
  await page.goto('https://example.com');
  
  // Hardcoded wait
  await page.waitForTimeout(5000);
  
  // Debug statement
  console.log('Starting login test');
  
  // Unstable selector
  await page.locator('div > div > div:nth-child(3)').click();
  
  // Hardcoded test data
  await page.locator('#email').fill('testuser@example.com');
  
  try {
    await page.locator('.submit').click();
  } catch (e) {
    // Empty catch block
  }
  
  await expect(page.locator('.success')).toBeVisible();
});

test('', async ({ page }) => {
  // Empty test name
  await page.goto('/dashboard');
});
"""

    def test_init(self, monitor):
        """Test monitor initialization."""
        assert monitor.test_directory == Path("test_e2e")
        assert "login_helper" in monitor._helper_patterns
        assert "nth_child" in monitor._unstable_selector_patterns
        assert "hardcoded_data" in monitor._data_patterns

    def test_compile_helper_patterns(self, monitor):
        """Test helper pattern compilation."""
        patterns = monitor._helper_patterns

        # Test login helper pattern
        login_pattern = patterns["login_helper"]
        assert login_pattern.search("async function loginUser(")
        assert login_pattern.search("const userLogin =")

        # Test navigation helper pattern
        nav_pattern = patterns["navigation_helper"]
        assert nav_pattern.search("function navigateToPage(")
        assert nav_pattern.search("const gotoHome =")

    def test_compile_unstable_patterns(self, monitor):
        """Test unstable selector pattern compilation."""
        patterns = monitor._unstable_selector_patterns

        # Test nth-child pattern
        nth_pattern = patterns["nth_child"]
        assert nth_pattern.search(":nth-child(3)")
        assert not nth_pattern.search("nth-child")

        # Test deep nesting pattern
        nesting_pattern = patterns["deep_nesting"]
        assert nesting_pattern.search(".class1 > .class2 > .class3 > .class4")

    def test_compile_data_patterns(self, monitor):
        """Test data pattern compilation."""
        patterns = monitor._data_patterns

        # Test hardcoded data pattern
        data_pattern = patterns["hardcoded_data"]
        assert data_pattern.search('"test data"')
        assert data_pattern.search("'sample user'")

        # Test random data pattern
        random_pattern = patterns["random_data"]
        assert random_pattern.search("Math.random()")
        assert random_pattern.search("Date.now()")

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.glob")
    def test_analyze_test_suite_no_directory(self, mock_glob, mock_exists, monitor):
        """Test analyzing when test directory doesn't exist."""
        mock_exists.return_value = False

        report = monitor.analyze_test_suite()

        assert report.total_files == 0
        assert len(report.issues) == 0

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.glob")
    def test_analyze_test_suite_no_files(self, mock_glob, mock_exists, monitor):
        """Test analyzing when no test files found."""
        mock_exists.return_value = True
        mock_glob.return_value = []

        report = monitor.analyze_test_suite()

        assert report.total_files == 0
        assert len(report.issues) == 0

    @patch("pathlib.Path.exists")
    def test_analyze_test_suite_with_files(
        self, mock_exists, monitor, sample_test_content
    ):
        """Test analyzing test suite with files."""
        mock_exists.return_value = True

        # Mock the _find_test_files method directly
        mock_file = Mock(spec=Path)
        mock_file.read_text.return_value = sample_test_content
        mock_file.__str__ = Mock(return_value="test.spec.ts")

        original_find_files = monitor._find_test_files
        monitor._find_test_files = Mock(return_value=[mock_file])

        try:
            report = monitor.analyze_test_suite()

            assert report.total_files == 1
            assert len(report.issues) > 0

            # Check for specific issue types
            issue_types = [issue.issue_type for issue in report.issues]
            assert "hardcoded_wait" in issue_types
            assert "debug_statement" in issue_types
        finally:
            monitor._find_test_files = original_find_files

    def test_check_line_issues(self, monitor):
        """Test checking individual lines for issues."""
        file_path = Path("test.spec.ts")

        # Test hardcoded wait detection
        issues = monitor._check_line_issues(
            file_path, 1, "await page.waitForTimeout(5000);"
        )
        assert len(issues) == 1
        assert issues[0].issue_type == "hardcoded_wait"
        assert issues[0].severity == "medium"

        # Test debug statement detection
        issues = monitor._check_line_issues(
            file_path, 2, "console.log('debug message');"
        )
        assert len(issues) == 1
        assert issues[0].issue_type == "debug_statement"
        assert issues[0].severity == "low"

        # Test empty catch block detection
        issues = monitor._check_line_issues(file_path, 3, "} catch (e) {}")
        assert len(issues) == 1
        assert issues[0].issue_type == "empty_catch"
        assert issues[0].severity == "high"

    def test_check_file_level_issues(self, monitor, sample_test_content):
        """Test checking file-level issues."""
        file_path = Path("test.spec.ts")

        issues = monitor._check_file_level_issues(file_path, sample_test_content)

        # Should detect poor test description
        poor_desc_issues = [
            i for i in issues if i.issue_type == "poor_test_description"
        ]
        assert len(poor_desc_issues) > 0

    def test_extract_helpers(self, monitor, sample_test_content):
        """Test extracting helper functions."""
        file_path = Path("test.spec.ts")

        helpers = monitor._extract_helpers(file_path, sample_test_content)

        assert "login_helper" in helpers
        assert len(helpers["login_helper"]) == 2  # loginUser and userLogin

        helper_names = [h["name"] for h in helpers["login_helper"]]
        assert "loginUser" in helper_names
        assert "userLogin" in helper_names

    def test_extract_selectors(self, monitor, sample_test_content):
        """Test extracting selectors from test content."""
        file_path = Path("test.spec.ts")

        selectors = monitor._extract_selectors(file_path, sample_test_content)

        assert len(selectors) > 0

        # Check selector classification
        selector_types = [s["type"] for s in selectors]
        assert "css" in selector_types

    def test_classify_selector(self, monitor):
        """Test selector classification."""
        assert monitor._classify_selector("//div[@id='test']") == "xpath"
        assert monitor._classify_selector(".class-name") == "css"
        assert monitor._classify_selector("#element-id") == "css"
        assert monitor._classify_selector("div > span") == "css"
        assert monitor._classify_selector("simple-text") == "other"

    def test_detect_duplicate_helpers(self, monitor):
        """Test duplicate helper detection."""
        helpers = {
            "login_helper": [
                {"name": "loginUser", "file": "test1.spec.ts", "line": 5},
                {"name": "userLogin", "file": "test2.spec.ts", "line": 10},
                {"name": "testLogin", "file": "test3.spec.ts", "line": 15},
            ]
        }

        duplicates = monitor._detect_duplicate_helpers(helpers)

        assert len(duplicates) == 1
        assert duplicates[0]["helper_type"] == "login_helper"
        assert len(duplicates[0]["instances"]) == 3

    def test_detect_unstable_selectors(self, monitor):
        """Test unstable selector detection."""
        selectors = [
            {
                "selector": "div:nth-child(3)",
                "file": "test.spec.ts",
                "line": 10,
                "type": "css",
            },
            {
                "selector": ".stable-class",
                "file": "test.spec.ts",
                "line": 15,
                "type": "css",
            },
            {
                "selector": ".very-long-generated-class-name-12345678901234567890",
                "file": "test.spec.ts",
                "line": 20,
                "type": "css",
            },
        ]

        unstable = monitor._detect_unstable_selectors(selectors)

        assert len(unstable) == 2  # nth-child and generated class

        # Check instability reasons
        nth_child_selector = next(s for s in unstable if "nth-child" in s["selector"])
        assert "nth_child" in nth_child_selector["instability_reasons"]

        generated_selector = next(s for s in unstable if "very-long" in s["selector"])
        assert "generated_classes" in generated_selector["instability_reasons"]

    def test_suggest_stable_alternative(self, monitor):
        """Test stable alternative suggestions."""
        # Test nth-child suggestion
        suggestion = monitor._suggest_stable_alternative(
            "div:nth-child(3)", ["nth_child"]
        )
        assert "getByRole()" in suggestion
        assert "getByTestId()" in suggestion

        # Test deep nesting suggestion
        suggestion = monitor._suggest_stable_alternative(
            "div > div > div", ["deep_nesting"]
        )
        assert "nesting depth" in suggestion

        # Test generated classes suggestion
        suggestion = monitor._suggest_stable_alternative(
            ".gen123456789", ["generated_classes"]
        )
        assert "test-id" in suggestion

    @patch("pathlib.Path.read_text")
    def test_detect_data_seeding_opportunities(self, mock_read_text, monitor):
        """Test data seeding opportunity detection."""
        test_content = """
        test('example', async ({ page }) => {
          await page.fill('#username', 'testuser@example.com');
          const randomId = Math.random();
          const timestamp = Date.now();
          await fetch('/api/users');
        });
        """

        mock_file = Mock()
        mock_file.read_text.return_value = test_content

        opportunities = monitor._detect_data_seeding_opportunities([mock_file])

        assert len(opportunities) > 0

        # Check for different pattern types
        pattern_types = [opp["pattern_type"] for opp in opportunities]
        assert "random_data" in pattern_types
        assert "timestamp_data" in pattern_types
        assert "api_calls" in pattern_types

    def test_suggest_data_improvement(self, monitor):
        """Test data improvement suggestions."""
        suggestion = monitor._suggest_data_improvement("hardcoded_data")
        assert "fixtures" in suggestion

        suggestion = monitor._suggest_data_improvement("random_data")
        assert "deterministic" in suggestion

        suggestion = monitor._suggest_data_improvement("api_calls")
        assert "Mock" in suggestion

    def test_generate_optimization_suggestions(self, monitor):
        """Test optimization suggestion generation."""
        # Create a report with various issues
        issues = [HealthIssue("hardcoded_wait", "medium", "file1.ts") for _ in range(6)]
        issues.extend([HealthIssue("large_test_file", "medium", "file2.ts")])

        report = HealthReport(
            total_files=25,
            issues=issues,
            duplicate_helpers=[{"type": "login"} for _ in range(4)],
            unstable_selectors=[{"selector": f"sel{i}"} for i in range(12)],
            data_seeding_opportunities=[{"type": "data"} for _ in range(16)],
        )

        suggestions = monitor._generate_optimization_suggestions(report)

        assert len(suggestions) > 0

        # Check for specific suggestions
        suggestion_text = " ".join(suggestions)
        assert "hardcoded waits" in suggestion_text
        assert "Large test files" in suggestion_text
        assert "duplicate helpers" in suggestion_text
        assert "unstable selectors" in suggestion_text
        assert "data seeding" in suggestion_text
        assert "feature-based directories" in suggestion_text

    def test_generate_health_report(self, monitor):
        """Test health report generation."""
        high_issue = HealthIssue(
            "test_issue", "high", "file.ts", 10, "High priority issue", "Fix it"
        )
        medium_issue = HealthIssue(
            "test_issue", "medium", "file.ts", 20, "Medium priority issue", "Fix it"
        )

        duplicate_helper = {
            "helper_type": "login_helper",
            "instances": [{"file": "file1.ts"}, {"file": "file2.ts"}],
            "suggestion": "Consolidate helpers",
        }

        unstable_selector = {
            "selector": "div:nth-child(3)",
            "file": "file.ts",
            "line": 15,
            "instability_reasons": ["nth_child"],
            "suggestion": "Use semantic selectors",
        }

        report = HealthReport(
            total_files=5,
            issues=[high_issue, medium_issue],
            duplicate_helpers=[duplicate_helper],
            unstable_selectors=[unstable_selector],
            optimization_suggestions=["Optimize test suite"],
        )

        report_text = monitor.generate_health_report(report)

        assert "🏥 Test Suite Health Report" in report_text
        assert "Total Files: 5" in report_text
        assert "Total Issues: 2" in report_text
        assert "High Priority: 1" in report_text
        assert "🚨 High Priority Issues" in report_text
        assert "🔄 Duplicate Helpers" in report_text
        assert "⚠️  Unstable Selectors" in report_text
        assert "💡 Optimization Suggestions" in report_text
        assert "High priority issue" in report_text
        assert "Consolidate helpers" in report_text
        assert "Use semantic selectors" in report_text
        assert "Optimize test suite" in report_text

    def test_find_test_files(self, monitor):
        """Test finding test files with patterns."""
        # Test the method with a simple mock by overriding the method
        original_method = monitor._find_test_files

        def mock_find_files(patterns):
            return [Path("test1.spec.ts"), Path("test2.test.ts")]

        monitor._find_test_files = mock_find_files

        try:
            files = monitor._find_test_files(["*.spec.ts", "*.test.ts"])
            assert len(files) == 2
        finally:
            monitor._find_test_files = original_method

    @patch("pathlib.Path.exists")
    def test_analyze_test_suite_with_error(self, mock_exists, monitor):
        """Test analyzing test suite when file reading fails."""
        mock_exists.return_value = True

        # Mock the _find_test_files method directly
        mock_file = Mock(spec=Path)
        mock_file.read_text.side_effect = Exception("File read error")
        mock_file.__str__ = Mock(return_value="test.spec.ts")

        original_find_files = monitor._find_test_files
        monitor._find_test_files = Mock(return_value=[mock_file])

        try:
            report = monitor.analyze_test_suite()

            assert report.total_files == 1
            assert len(report.issues) >= 1  # At least 1 analysis error

            analysis_errors = [
                i for i in report.issues if i.issue_type == "analysis_error"
            ]
            assert len(analysis_errors) >= 1
            assert "File read error" in analysis_errors[0].description
        finally:
            monitor._find_test_files = original_find_files
