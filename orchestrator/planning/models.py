"""
Data models for test planning and specification.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class Priority(Enum):
    """Test priority levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TestStatus(Enum):
    """Test execution status."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TestSpecification:
    """Test specification input from user."""
    
    id: str
    name: str
    description: str
    requirements: List[str]
    priority: Priority = Priority.MEDIUM
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "requirements": self.requirements,
            "priority": self.priority.value,
            "tags": self.tags,
            "metadata": self.metadata
        }


@dataclass
class TestStep:
    """Individual test step."""
    
    action: str
    target: str
    value: Optional[str] = None
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "action": self.action,
            "target": self.target,
            "value": self.value,
            "description": self.description
        }


@dataclass
class Assertion:
    """Test assertion."""
    
    type: str
    target: str
    expected: str
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type,
            "target": self.target,
            "expected": self.expected,
            "description": self.description
        }


@dataclass
class SelectorInfo:
    """Information about selectors used in test."""
    
    selector: str
    type: str  # 'role', 'label', 'testid', 'css', etc.
    element_description: str
    is_compliant: bool = True
    justification: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "selector": self.selector,
            "type": self.type,
            "element_description": self.element_description,
            "is_compliant": self.is_compliant,
            "justification": self.justification
        }


@dataclass
class PageObject:
    """Page object definition."""
    
    name: str
    url_pattern: str
    selectors: List[SelectorInfo]
    methods: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "url_pattern": self.url_pattern,
            "selectors": [s.to_dict() for s in self.selectors],
            "methods": self.methods
        }


@dataclass
class TestCase:
    """Individual test case."""
    
    name: str
    description: str
    steps: List[TestStep]
    assertions: List[Assertion]
    selectors: List[SelectorInfo] = field(default_factory=list)
    setup_requirements: List[str] = field(default_factory=list)
    estimated_duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "assertions": [a.to_dict() for a in self.assertions],
            "selectors": [s.to_dict() for s in self.selectors],
            "setup_requirements": self.setup_requirements,
            "estimated_duration": self.estimated_duration
        }


@dataclass
class TestPlan:
    """Complete test plan generated from specification."""
    
    specification_id: str
    test_cases: List[TestCase]
    page_objects: List[PageObject] = field(default_factory=list)
    setup_requirements: List[str] = field(default_factory=list)
    estimated_duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "specification_id": self.specification_id,
            "test_cases": [tc.to_dict() for tc in self.test_cases],
            "page_objects": [po.to_dict() for po in self.page_objects],
            "setup_requirements": self.setup_requirements,
            "estimated_duration": self.estimated_duration,
            "metadata": self.metadata
        }


@dataclass
class TestGapAnalysis:
    """Analysis of gaps in test coverage."""
    
    missing_test_cases: List[str]
    incomplete_coverage: List[str]
    suggested_additions: List[str]
    priority_gaps: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "missing_test_cases": self.missing_test_cases,
            "incomplete_coverage": self.incomplete_coverage,
            "suggested_additions": self.suggested_additions,
            "priority_gaps": self.priority_gaps
        }