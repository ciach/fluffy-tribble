"""
Test suite health monitoring for detecting and suggesting improvements.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


@dataclass
class HealthIssue:
    """Represents a test suite health issue."""

    issue_type: str
    severity: str  # 'high', 'medium', 'low'
    file_path: str
    line_number: Optional[int] = None
    description: str = ""
    suggestion: str = ""
    affected_code: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "issue_type": self.issue_type,
            "severity": self.severity,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "description": self.description,
            "suggestion": self.suggestion,
            "affected_code": self.affected_code,
            "metadata": self.metadata,
        }


@dataclass
class HealthReport:
    """Comprehensive test suite health report."""

    total_files: int = 0
    issues: List[HealthIssue] = field(default_factory=list)
    duplicate_helpers: List[Dict[str, Any]] = field(default_factory=list)
    unstable_selectors: List[Dict[str, Any]] = field(default_factory=list)
    data_seeding_opportunities: List[Dict[str, Any]] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)

    @property
    def high_priority_issues(self) -> List[HealthIssue]:
        """Get high priority issues."""
        return [issue for issue in self.issues if issue.severity == "high"]

    @property
    def medium_priority_issues(self) -> List[HealthIssue]:
        """Get medium priority issues."""
        return [issue for issue in self.issues if issue.severity == "medium"]

    @property
    def low_priority_issues(self) -> List[HealthIssue]:
        """Get low priority issues."""
        return [issue for issue in self.issues if issue.severity == "low"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_files": self.total_files,
            "total_issues": len(self.issues),
            "high_priority_count": len(self.high_priority_issues),
            "medium_priority_count": len(self.medium_priority_issues),
            "low_priority_count": len(self.low_priority_issues),
            "issues": [issue.to_dict() for issue in self.issues],
            "duplicate_helpers": self.duplicate_helpers,
            "unstable_selectors": self.unstable_selectors,
            "data_seeding_opportunities": self.data_seeding_opportunities,
            "optimization_suggestions": self.optimization_suggestions,
        }


class TestSuiteHealthMonitor:
    """Monitors test suite health and suggests improvements."""

    def __init__(self, test_directory: str = "e2e"):
        """Initialize the health monitor.

        Args:
            test_directory: Directory containing test files
        """
        self.test_directory = Path(test_directory)
        self._helper_patterns = self._compile_helper_patterns()
        self._unstable_selector_patterns = self._compile_unstable_patterns()
        self._data_patterns = self._compile_data_patterns()

    def _compile_helper_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for detecting helper functions."""
        return {
            "login_helper": re.compile(
                r"(?:async\s+)?function\s+(\w*login\w*)\s*\(|const\s+(\w*login\w*)\s*=",
                re.IGNORECASE,
            ),
            "navigation_helper": re.compile(
                r"(?:async\s+)?function\s+(\w*(?:navigate|goto|visit)\w*)\s*\(|const\s+(\w*(?:navigate|goto|visit)\w*)\s*=",
                re.IGNORECASE,
            ),
            "form_helper": re.compile(
                r"(?:async\s+)?function\s+(\w*(?:fill|submit|form)\w*)\s*\(|const\s+(\w*(?:fill|submit|form)\w*)\s*=",
                re.IGNORECASE,
            ),
            "wait_helper": re.compile(
                r"(?:async\s+)?function\s+(\w*wait\w*)\s*\(|const\s+(\w*wait\w*)\s*=",
                re.IGNORECASE,
            ),
            "setup_helper": re.compile(
                r"(?:async\s+)?function\s+(\w*(?:setup|prepare|init)\w*)\s*\(|const\s+(\w*(?:setup|prepare|init)\w*)\s*=",
                re.IGNORECASE,
            ),
        }

    def _compile_unstable_patterns(self) -> Dict[str, re.Pattern]:
        """Compile patterns for detecting unstable selectors."""
        return {
            "nth_child": re.compile(r":nth-child\(\d+\)"),
            "nth_of_type": re.compile(r":nth-of-type\(\d+\)"),
            "deep_nesting": re.compile(r"(?:[.#]\w+\s*>\s*){3,}"),
            "generated_classes": re.compile(r"\.[a-zA-Z0-9-_]{15,}"),
            "generated_ids": re.compile(r"#[a-zA-Z0-9-_]{15,}"),
            "position_dependent": re.compile(r"(?:first|last|eq\(\d+\))"),
            "text_content": re.compile(r"contains\(['\"][^'\"]*['\"]"),
        }

    def _compile_data_patterns(self) -> Dict[str, re.Pattern]:
        """Compile patterns for detecting data seeding opportunities."""
        return {
            "hardcoded_data": re.compile(
                r"['\"](?:test|demo|sample|example)[\w\s]*['\"]", re.IGNORECASE
            ),
            "random_data": re.compile(r"Math\.random\(\)|Date\.now\(\)|uuid\(\)"),
            "timestamp_data": re.compile(r"new Date\(\)|Date\.now\(\)"),
            "user_data": re.compile(
                r"['\"](?:test|demo|sample)(?:user|admin)(?:\d+|@example\.com)['\"]",
                re.IGNORECASE,
            ),
            "api_calls": re.compile(r"(?:fetch|axios|request)\s*\("),
        }

    def analyze_test_suite(self, file_patterns: List[str] = None) -> HealthReport:
        """Analyze the entire test suite for health issues.

        Args:
            file_patterns: Optional list of file patterns to analyze

        Returns:
            Comprehensive health report
        """
        logger.info(f"Starting test suite health analysis in {self.test_directory}")

        if not self.test_directory.exists():
            logger.warning(f"Test directory not found: {self.test_directory}")
            return HealthReport()

        # Find test files
        test_files = self._find_test_files(file_patterns or ["*.spec.ts", "*.test.ts"])

        if not test_files:
            logger.warning("No test files found")
            return HealthReport()

        report = HealthReport(total_files=len(test_files))

        # Analyze each file
        all_helpers = defaultdict(list)
        all_selectors = []

        for file_path in test_files:
            try:
                file_content = file_path.read_text()

                # Detect issues in this file
                file_issues = self._analyze_file(file_path, file_content)
                report.issues.extend(file_issues)

                # Collect helpers for duplicate detection
                helpers = self._extract_helpers(file_path, file_content)
                for helper_type, helper_list in helpers.items():
                    all_helpers[helper_type].extend(helper_list)

                # Collect selectors for stability analysis
                selectors = self._extract_selectors(file_path, file_content)
                all_selectors.extend(selectors)

            except Exception as e:
                logger.error(f"Failed to analyze file {file_path}: {e}")
                report.issues.append(
                    HealthIssue(
                        issue_type="analysis_error",
                        severity="medium",
                        file_path=str(file_path),
                        description=f"Failed to analyze file: {e}",
                        suggestion="Check file syntax and encoding",
                    )
                )

        # Analyze for duplicates and patterns
        report.duplicate_helpers = self._detect_duplicate_helpers(all_helpers)
        report.unstable_selectors = self._detect_unstable_selectors(all_selectors)
        report.data_seeding_opportunities = self._detect_data_seeding_opportunities(
            test_files
        )
        report.optimization_suggestions = self._generate_optimization_suggestions(
            report
        )

        logger.info(f"Health analysis completed: {len(report.issues)} issues found")
        return report

    def _find_test_files(self, patterns: List[str]) -> List[Path]:
        """Find test files matching the given patterns."""
        test_files = []
        for pattern in patterns:
            test_files.extend(self.test_directory.glob(f"**/{pattern}"))
        return sorted(test_files, key=str)

    def _analyze_file(self, file_path: Path, content: str) -> List[HealthIssue]:
        """Analyze a single test file for health issues."""
        issues = []
        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            # Check for common anti-patterns
            issues.extend(self._check_line_issues(file_path, line_num, line))

        # Check file-level issues
        issues.extend(self._check_file_level_issues(file_path, content))

        return issues

    def _check_line_issues(
        self, file_path: Path, line_num: int, line: str
    ) -> List[HealthIssue]:
        """Check a single line for health issues."""
        issues = []

        # Check for hardcoded waits
        if re.search(r"(?:sleep|waitForTimeout)\s*\(\s*\d+", line):
            issues.append(
                HealthIssue(
                    issue_type="hardcoded_wait",
                    severity="medium",
                    file_path=str(file_path),
                    line_number=line_num,
                    description="Hardcoded wait/sleep found",
                    suggestion="Use expect().toBeVisible() or other assertion-based waits",
                    affected_code=line.strip(),
                )
            )

        # Check for console.log statements
        if re.search(r"console\.log\s*\(", line):
            issues.append(
                HealthIssue(
                    issue_type="debug_statement",
                    severity="low",
                    file_path=str(file_path),
                    line_number=line_num,
                    description="Debug console.log statement found",
                    suggestion="Remove debug statements before committing",
                    affected_code=line.strip(),
                )
            )

        # Check for try-catch without proper error handling
        if re.search(r"catch\s*\(\s*\w*\s*\)\s*\{\s*\}", line):
            issues.append(
                HealthIssue(
                    issue_type="empty_catch",
                    severity="high",
                    file_path=str(file_path),
                    line_number=line_num,
                    description="Empty catch block found",
                    suggestion="Add proper error handling or logging in catch blocks",
                    affected_code=line.strip(),
                )
            )

        return issues

    def _check_file_level_issues(
        self, file_path: Path, content: str
    ) -> List[HealthIssue]:
        """Check file-level health issues."""
        issues = []

        # Check for missing test descriptions
        test_blocks = re.findall(r"test\s*\(\s*['\"]([^'\"]*)['\"]", content)
        for test_name in test_blocks:
            if not test_name.strip() or len(test_name) < 5:
                issues.append(
                    HealthIssue(
                        issue_type="poor_test_description",
                        severity="medium",
                        file_path=str(file_path),
                        description=f"Test has poor description: '{test_name}'",
                        suggestion="Use descriptive test names that explain what is being tested",
                    )
                )

        # Check for excessive test file size
        line_count = len(content.split("\n"))
        if line_count > 500:
            issues.append(
                HealthIssue(
                    issue_type="large_test_file",
                    severity="medium",
                    file_path=str(file_path),
                    description=f"Test file is very large ({line_count} lines)",
                    suggestion="Consider splitting large test files into smaller, focused files",
                    metadata={"line_count": line_count},
                )
            )

        return issues

    def _extract_helpers(
        self, file_path: Path, content: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Extract helper functions from a test file."""
        helpers = defaultdict(list)

        for helper_type, pattern in self._helper_patterns.items():
            matches = pattern.finditer(content)
            for match in matches:
                helper_name = match.group(1) or match.group(2)
                if helper_name:
                    helpers[helper_type].append(
                        {
                            "name": helper_name,
                            "file": str(file_path),
                            "line": content[: match.start()].count("\n") + 1,
                            "full_match": match.group(0),
                        }
                    )

        return helpers

    def _extract_selectors(self, file_path: Path, content: str) -> List[Dict[str, Any]]:
        """Extract selectors from a test file."""
        selectors = []

        # Find locator calls
        locator_pattern = re.compile(r"locator\s*\(\s*['\"]([^'\"]+)['\"]")
        matches = locator_pattern.finditer(content)

        for match in matches:
            selector = match.group(1)
            selectors.append(
                {
                    "selector": selector,
                    "file": str(file_path),
                    "line": content[: match.start()].count("\n") + 1,
                    "type": self._classify_selector(selector),
                }
            )

        return selectors

    def _classify_selector(self, selector: str) -> str:
        """Classify a selector by type."""
        if selector.startswith("//"):
            return "xpath"
        elif any(char in selector for char in [".", "#", ">", "+"]):
            return "css"
        else:
            return "other"

    def _detect_duplicate_helpers(
        self, all_helpers: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Detect duplicate helper functions across files."""
        duplicates = []

        for helper_type, helpers in all_helpers.items():
            if len(helpers) < 2:
                continue

            # Group by similar names (fuzzy matching)
            name_groups = defaultdict(list)
            for helper in helpers:
                # Simple similarity: remove common prefixes/suffixes and normalize
                normalized_name = re.sub(
                    r"^(test|spec|helper|util)", "", helper["name"], flags=re.IGNORECASE
                )
                normalized_name = re.sub(
                    r"(test|spec|helper|util)$",
                    "",
                    normalized_name,
                    flags=re.IGNORECASE,
                )
                # Further normalize by removing common variations
                normalized_name = re.sub(
                    r"(user|test)", "", normalized_name, flags=re.IGNORECASE
                )
                # Normalize login variations
                if "login" in normalized_name.lower():
                    normalized_name = "login"
                name_groups[normalized_name.lower()].append(helper)

            # Find groups with multiple helpers
            for normalized_name, group in name_groups.items():
                if len(group) > 1:
                    duplicates.append(
                        {
                            "helper_type": helper_type,
                            "normalized_name": normalized_name,
                            "instances": group,
                            "suggestion": f"Consider consolidating {len(group)} similar {helper_type} helpers into a shared utility",
                        }
                    )

        return duplicates

    def _detect_unstable_selectors(
        self, all_selectors: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect potentially unstable selectors."""
        unstable = []

        for selector_info in all_selectors:
            selector = selector_info["selector"]
            instability_reasons = []

            # Check against unstable patterns
            for pattern_name, pattern in self._unstable_selector_patterns.items():
                if pattern.search(selector):
                    instability_reasons.append(pattern_name)

            if instability_reasons:
                unstable.append(
                    {
                        **selector_info,
                        "instability_reasons": instability_reasons,
                        "suggestion": self._suggest_stable_alternative(
                            selector, instability_reasons
                        ),
                    }
                )

        return unstable

    def _suggest_stable_alternative(self, selector: str, reasons: List[str]) -> str:
        """Suggest a more stable alternative for an unstable selector."""
        suggestions = []

        if "nth_child" in reasons or "nth_of_type" in reasons:
            suggestions.append(
                "Use getByRole() or getByTestId() instead of positional selectors"
            )

        if "deep_nesting" in reasons:
            suggestions.append(
                "Reduce selector nesting depth, use more specific attributes"
            )

        if "generated_classes" in reasons or "generated_ids" in reasons:
            suggestions.append(
                "Use semantic selectors or add stable test-id attributes"
            )

        if "text_content" in reasons:
            suggestions.append(
                "Use getByRole() with accessible name instead of text content matching"
            )

        return (
            "; ".join(suggestions)
            if suggestions
            else "Consider using semantic selectors like getByRole(), getByLabel(), or getByTestId()"
        )

    def _detect_data_seeding_opportunities(
        self, test_files: List[Path]
    ) -> List[Dict[str, Any]]:
        """Detect opportunities for deterministic data seeding."""
        opportunities = []

        for file_path in test_files:
            try:
                content = file_path.read_text()

                # Check for hardcoded test data
                for pattern_name, pattern in self._data_patterns.items():
                    matches = pattern.finditer(content)
                    for match in matches:
                        opportunities.append(
                            {
                                "file": str(file_path),
                                "line": content[: match.start()].count("\n") + 1,
                                "pattern_type": pattern_name,
                                "matched_text": match.group(0),
                                "suggestion": self._suggest_data_improvement(
                                    pattern_name
                                ),
                            }
                        )

            except Exception as e:
                logger.error(f"Failed to analyze data patterns in {file_path}: {e}")

        return opportunities

    def _suggest_data_improvement(self, pattern_type: str) -> str:
        """Suggest improvements for data handling patterns."""
        suggestions = {
            "hardcoded_data": "Use test fixtures or data factories for consistent test data",
            "random_data": "Use deterministic data generation with fixed seeds for reproducible tests",
            "timestamp_data": "Use fixed timestamps or relative dates for predictable test behavior",
            "user_data": "Create reusable user fixtures with consistent test accounts",
            "api_calls": "Mock API calls or use test-specific endpoints for reliable testing",
        }
        return suggestions.get(
            pattern_type, "Consider using more deterministic data patterns"
        )

    def _generate_optimization_suggestions(self, report: HealthReport) -> List[str]:
        """Generate overall optimization suggestions based on the health report."""
        suggestions = []

        # Analyze issue patterns
        issue_types = Counter(issue.issue_type for issue in report.issues)

        if issue_types.get("hardcoded_wait", 0) > 5:
            suggestions.append(
                "High number of hardcoded waits detected. Consider implementing a wait utility with assertion-based waits."
            )

        if issue_types.get("large_test_file", 0) > 0:
            suggestions.append(
                "Large test files detected. Consider splitting tests by feature or user journey."
            )

        if len(report.duplicate_helpers) > 3:
            suggestions.append(
                "Multiple duplicate helpers found. Consider creating a shared test utilities library."
            )

        if len(report.unstable_selectors) > 10:
            suggestions.append(
                "Many unstable selectors detected. Implement a selector strategy guide and review existing selectors."
            )

        if len(report.data_seeding_opportunities) > 15:
            suggestions.append(
                "Significant data seeding opportunities found. Implement test data factories and fixtures."
            )

        # File organization suggestions
        if report.total_files > 20:
            suggestions.append(
                "Large test suite detected. Consider organizing tests into feature-based directories."
            )

        return suggestions

    def generate_health_report(self, report: HealthReport) -> str:
        """Generate a formatted health report.

        Args:
            report: Health report data

        Returns:
            Formatted health report string
        """
        lines = []
        lines.append("🏥 Test Suite Health Report")
        lines.append("=" * 50)
        lines.append(f"📊 Summary:")
        lines.append(f"   Total Files: {report.total_files}")
        lines.append(f"   Total Issues: {len(report.issues)}")
        lines.append(f"   High Priority: {len(report.high_priority_issues)}")
        lines.append(f"   Medium Priority: {len(report.medium_priority_issues)}")
        lines.append(f"   Low Priority: {len(report.low_priority_issues)}")
        lines.append("")

        # High priority issues
        if report.high_priority_issues:
            lines.append("🚨 High Priority Issues:")
            for issue in report.high_priority_issues[:10]:  # Limit to top 10
                lines.append(f"   • {issue.description}")
                lines.append(
                    f"     File: {issue.file_path}:{issue.line_number or 'N/A'}"
                )
                lines.append(f"     Fix: {issue.suggestion}")
                lines.append("")

        # Duplicate helpers
        if report.duplicate_helpers:
            lines.append("🔄 Duplicate Helpers:")
            for dup in report.duplicate_helpers[:5]:  # Limit to top 5
                lines.append(
                    f"   • {dup['helper_type']}: {len(dup['instances'])} similar helpers"
                )
                lines.append(
                    f"     Files: {', '.join(set(h['file'] for h in dup['instances']))}"
                )
                lines.append(f"     Suggestion: {dup['suggestion']}")
                lines.append("")

        # Unstable selectors
        if report.unstable_selectors:
            lines.append("⚠️  Unstable Selectors:")
            for sel in report.unstable_selectors[:5]:  # Limit to top 5
                lines.append(f"   • {sel['selector']}")
                lines.append(f"     File: {sel['file']}:{sel['line']}")
                lines.append(f"     Issues: {', '.join(sel['instability_reasons'])}")
                lines.append(f"     Suggestion: {sel['suggestion']}")
                lines.append("")

        # Optimization suggestions
        if report.optimization_suggestions:
            lines.append("💡 Optimization Suggestions:")
            for suggestion in report.optimization_suggestions:
                lines.append(f"   • {suggestion}")
            lines.append("")

        return "\n".join(lines)
