"""
Data models for test generation and selector auditing.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class SelectorType(Enum):
    """Types of selectors used in tests."""

    ROLE = "role"
    LABEL = "label"
    TEST_ID = "testid"
    PLACEHOLDER = "placeholder"
    TEXT = "text"
    CSS = "css"
    XPATH = "xpath"
    OTHER = "other"


class ViolationSeverity(Enum):
    """Severity levels for selector policy violations."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class SelectorViolation:
    """Represents a selector policy violation."""

    line_number: int
    selector: str
    selector_type: SelectorType
    violation_type: str
    message: str
    severity: ViolationSeverity
    suggested_fix: Optional[str] = None
    justification: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "line_number": self.line_number,
            "selector": self.selector,
            "selector_type": self.selector_type.value,
            "violation_type": self.violation_type,
            "message": self.message,
            "severity": self.severity.value,
            "suggested_fix": self.suggested_fix,
            "justification": self.justification,
        }


@dataclass
class AuditResult:
    """Result of selector audit."""

    is_compliant: bool
    violations: List[SelectorViolation] = field(default_factory=list)
    total_selectors: int = 0
    compliant_selectors: int = 0
    justified_violations: int = 0

    @property
    def compliance_rate(self) -> float:
        """Calculate compliance rate as percentage."""
        if self.total_selectors == 0:
            return 100.0
        return (self.compliant_selectors / self.total_selectors) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_compliant": self.is_compliant,
            "violations": [v.to_dict() for v in self.violations],
            "total_selectors": self.total_selectors,
            "compliant_selectors": self.compliant_selectors,
            "justified_violations": self.justified_violations,
            "compliance_rate": self.compliance_rate,
        }


@dataclass
class GeneratedTest:
    """Represents a generated test file."""

    name: str
    content: str
    file_path: str
    test_plan_id: str
    page_objects: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    audit_result: Optional[AuditResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "content_length": len(self.content),
            "file_path": self.file_path,
            "test_plan_id": self.test_plan_id,
            "page_objects": self.page_objects,
            "imports": self.imports,
            "audit_result": self.audit_result.to_dict() if self.audit_result else None,
            "metadata": self.metadata,
        }


@dataclass
class PageObjectScaffold:
    """Scaffold for page object pattern."""

    class_name: str
    file_path: str
    url_pattern: str
    selectors: Dict[str, str] = field(default_factory=dict)
    methods: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "class_name": self.class_name,
            "file_path": self.file_path,
            "url_pattern": self.url_pattern,
            "selectors": self.selectors,
            "methods": self.methods,
            "imports": self.imports,
        }
