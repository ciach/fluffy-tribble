"""
Data models for failure analysis and patching.

Defines Pydantic models for failure analysis results, fix suggestions,
and patch operations.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from pydantic import BaseModel, Field, validator, ConfigDict


class ErrorCategory(Enum):
    """Categories of test failure errors."""

    SELECTOR = "selector"
    TIMING = "timing"
    NETWORK = "network"
    ASSERTION = "assertion"
    NAVIGATION = "navigation"
    ELEMENT_STATE = "element_state"
    TIMEOUT = "timeout"
    BROWSER_CRASH = "browser_crash"
    OTHER = "other"


class ConfidenceLevel(Enum):
    """Confidence levels for analysis and fixes."""

    HIGH = "high"  # 80-100% confidence
    MEDIUM = "medium"  # 50-79% confidence
    LOW = "low"  # 0-49% confidence


class FixType(Enum):
    """Types of fixes that can be applied."""

    SELECTOR_REPLACEMENT = "selector_replacement"
    WAIT_CONDITION = "wait_condition"
    TIMEOUT_ADJUSTMENT = "timeout_adjustment"
    ASSERTION_UPDATE = "assertion_update"
    NAVIGATION_FIX = "navigation_fix"
    ELEMENT_INTERACTION = "element_interaction"
    ERROR_HANDLING = "error_handling"
    RETRY_LOGIC = "retry_logic"


class ArtifactAnalysis(BaseModel):
    """Analysis of a specific test artifact."""

    model_config = ConfigDict(extra="forbid")

    artifact_path: str = Field(..., description="Path to the analyzed artifact")
    artifact_type: str = Field(
        ..., description="Type of artifact (trace, screenshot, etc.)"
    )
    findings: List[str] = Field(
        default_factory=list, description="Key findings from this artifact"
    )
    error_indicators: List[str] = Field(
        default_factory=list, description="Error indicators found"
    )
    relevant_sections: List[Dict[str, Any]] = Field(
        default_factory=list, description="Relevant sections or timestamps"
    )

    @validator("artifact_path")
    def validate_artifact_path(cls, v):
        """Validate artifact path exists."""
        if not Path(v).exists():
            raise ValueError(f"Artifact file does not exist: {v}")
        return v


class StackTraceAnalysis(BaseModel):
    """Analysis of error stack trace."""

    model_config = ConfigDict(extra="forbid")

    raw_stack_trace: str = Field(..., description="Original stack trace")
    error_message: str = Field(..., description="Primary error message")
    error_type: str = Field(..., description="Type of error (e.g., TimeoutError)")
    failing_line: Optional[int] = Field(
        None, description="Line number where error occurred"
    )
    failing_code: Optional[str] = Field(
        None, description="Code that caused the failure"
    )
    playwright_action: Optional[str] = Field(
        None, description="Playwright action that failed"
    )
    selector_used: Optional[str] = Field(None, description="Selector that was used")

    @validator("raw_stack_trace")
    def validate_stack_trace(cls, v):
        """Validate stack trace is not empty."""
        if not v.strip():
            raise ValueError("Stack trace cannot be empty")
        return v


class FixSuggestion(BaseModel):
    """A suggested fix for a test failure."""

    model_config = ConfigDict(extra="forbid")

    fix_type: FixType = Field(..., description="Type of fix being suggested")
    description: str = Field(..., description="Human-readable description of the fix")
    confidence: ConfidenceLevel = Field(..., description="Confidence level in this fix")

    # Code changes
    original_code: Optional[str] = Field(
        None, description="Original code to be replaced"
    )
    suggested_code: Optional[str] = Field(
        None, description="Suggested replacement code"
    )
    line_number: Optional[int] = Field(None, description="Line number to modify")

    # Additional context
    reasoning: str = Field(..., description="Explanation of why this fix should work")
    potential_side_effects: List[str] = Field(
        default_factory=list, description="Potential negative impacts of this fix"
    )
    test_impact: str = Field(
        default="minimal", description="Expected impact on test behavior"
    )

    # Validation
    requires_manual_review: bool = Field(
        default=False,
        description="Whether this fix requires manual review before applying",
    )

    @validator("confidence")
    def validate_confidence_requirements(cls, v, values):
        """Validate confidence level requirements."""
        fix_type = values.get("fix_type")

        # High-risk fixes should require manual review
        high_risk_types = [
            FixType.ASSERTION_UPDATE,
            FixType.ERROR_HANDLING,
            FixType.RETRY_LOGIC,
        ]

        if fix_type in high_risk_types and v == ConfidenceLevel.HIGH:
            values["requires_manual_review"] = True

        return v


class FailureAnalysis(BaseModel):
    """Complete analysis of a test failure."""

    model_config = ConfigDict(extra="forbid")

    # Test identification
    test_name: str = Field(..., description="Name of the failed test")
    test_file: str = Field(..., description="Path to the test file")
    workflow_id: str = Field(..., description="Workflow ID for correlation")

    # Analysis metadata
    analysis_id: str = Field(..., description="Unique ID for this analysis")
    analyzed_at: datetime = Field(
        default_factory=datetime.utcnow, description="When analysis was performed"
    )
    analysis_duration: float = Field(
        ..., ge=0, description="Time taken for analysis in seconds"
    )

    # Error categorization
    primary_error_category: ErrorCategory = Field(
        ..., description="Primary error category"
    )
    secondary_categories: List[ErrorCategory] = Field(
        default_factory=list, description="Additional error categories"
    )

    # Analysis components
    stack_trace_analysis: Optional[StackTraceAnalysis] = Field(
        None, description="Analysis of the stack trace"
    )
    artifact_analyses: List[ArtifactAnalysis] = Field(
        default_factory=list, description="Analysis of test artifacts"
    )

    # Root cause analysis
    root_cause: str = Field(..., description="Identified root cause of the failure")
    contributing_factors: List[str] = Field(
        default_factory=list,
        description="Additional factors that contributed to failure",
    )

    # Fix suggestions
    fix_suggestions: List[FixSuggestion] = Field(
        default_factory=list, description="Suggested fixes ordered by confidence"
    )

    # Analysis confidence
    overall_confidence: ConfidenceLevel = Field(
        ..., description="Overall confidence in the analysis"
    )

    # Additional context
    similar_failures: List[str] = Field(
        default_factory=list, description="Similar failures seen in the past"
    )
    environment_factors: Dict[str, Any] = Field(
        default_factory=dict,
        description="Environment factors that may have contributed",
    )

    @validator("fix_suggestions")
    def validate_fix_suggestions_order(cls, v):
        """Ensure fix suggestions are ordered by confidence (highest first)."""
        if not v:
            return v

        # Sort by confidence level (HIGH > MEDIUM > LOW)
        confidence_order = {
            ConfidenceLevel.HIGH: 3,
            ConfidenceLevel.MEDIUM: 2,
            ConfidenceLevel.LOW: 1,
        }

        sorted_suggestions = sorted(
            v, key=lambda x: confidence_order[x.confidence], reverse=True
        )

        return sorted_suggestions

    @property
    def has_high_confidence_fixes(self) -> bool:
        """Check if there are any high-confidence fix suggestions."""
        return any(
            fix.confidence == ConfidenceLevel.HIGH for fix in self.fix_suggestions
        )

    @property
    def recommended_fix(self) -> Optional[FixSuggestion]:
        """Get the highest confidence fix suggestion."""
        return self.fix_suggestions[0] if self.fix_suggestions else None

    def get_fixes_by_type(self, fix_type: FixType) -> List[FixSuggestion]:
        """Get all fix suggestions of a specific type."""
        return [fix for fix in self.fix_suggestions if fix.fix_type == fix_type]

    def to_summary(self) -> Dict[str, Any]:
        """Create a summary dictionary for logging."""
        return {
            "test_name": self.test_name,
            "analysis_id": self.analysis_id,
            "primary_error_category": self.primary_error_category.value,
            "root_cause": self.root_cause,
            "fix_suggestions_count": len(self.fix_suggestions),
            "has_high_confidence_fixes": self.has_high_confidence_fixes,
            "overall_confidence": self.overall_confidence.value,
            "analysis_duration": self.analysis_duration,
            "workflow_id": self.workflow_id,
        }


class FailureAnalysisError(Exception):
    """Exception raised during failure analysis."""

    def __init__(
        self,
        message: str,
        test_name: Optional[str] = None,
        analysis_id: Optional[str] = None,
        error_category: Optional[ErrorCategory] = None,
    ):
        super().__init__(message)
        self.test_name = test_name
        self.analysis_id = analysis_id
        self.error_category = error_category

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            "error": str(self),
            "test_name": self.test_name,
            "analysis_id": self.analysis_id,
            "error_category": (
                self.error_category.value if self.error_category else None
            ),
        }


class PatchOperation(Enum):
    """Types of patch operations."""

    REPLACE_LINE = "replace_line"
    REPLACE_BLOCK = "replace_block"
    INSERT_BEFORE = "insert_before"
    INSERT_AFTER = "insert_after"
    DELETE_LINE = "delete_line"
    DELETE_BLOCK = "delete_block"


class PatchValidationResult(BaseModel):
    """Result of patch validation."""

    model_config = ConfigDict(extra="forbid")

    is_valid: bool = Field(..., description="Whether the patch is valid")
    validation_errors: List[str] = Field(
        default_factory=list, description="List of validation errors"
    )
    warnings: List[str] = Field(
        default_factory=list, description="List of validation warnings"
    )
    syntax_valid: bool = Field(..., description="Whether syntax is valid after patch")
    semantic_changes: List[str] = Field(
        default_factory=list, description="Semantic changes detected"
    )

    @property
    def has_errors(self) -> bool:
        """Check if validation has errors."""
        return len(self.validation_errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return len(self.warnings) > 0


class CodePatch(BaseModel):
    """A code patch to be applied to a test file."""

    model_config = ConfigDict(extra="forbid")

    # Patch identification
    patch_id: str = Field(..., description="Unique patch identifier")
    fix_suggestion_id: Optional[str] = Field(
        None, description="ID of the fix suggestion this patch implements"
    )

    # Target file information
    target_file: str = Field(..., description="Path to file to be patched")
    original_content: str = Field(..., description="Original file content")

    # Patch operation
    operation: PatchOperation = Field(..., description="Type of patch operation")
    start_line: int = Field(..., ge=1, description="Starting line number (1-based)")
    end_line: Optional[int] = Field(
        None, ge=1, description="Ending line number for block operations"
    )

    # Patch content
    original_code: str = Field(..., description="Original code to be replaced")
    patched_code: str = Field(..., description="New code to replace original")

    # Patch metadata
    description: str = Field(..., description="Description of what the patch does")
    reasoning: str = Field(..., description="Why this patch is needed")
    confidence: ConfidenceLevel = Field(..., description="Confidence in this patch")

    # Validation and safety
    requires_backup: bool = Field(
        default=True, description="Whether to create backup before applying"
    )
    requires_validation: bool = Field(
        default=True, description="Whether to validate after applying"
    )
    dry_run_tested: bool = Field(
        default=False, description="Whether patch was tested in dry run"
    )

    # Application tracking
    applied_at: Optional[datetime] = Field(None, description="When patch was applied")
    applied_by: Optional[str] = Field(None, description="Who/what applied the patch")
    backup_path: Optional[str] = Field(None, description="Path to backup file")

    @validator("end_line")
    def validate_line_range(cls, v, values):
        """Validate that end_line is after start_line for block operations."""
        start_line = values.get("start_line")
        operation = values.get("operation")

        if operation in [PatchOperation.REPLACE_BLOCK, PatchOperation.DELETE_BLOCK]:
            if v is None:
                raise ValueError(
                    f"end_line is required for {operation.value} operations"
                )
            if start_line and v <= start_line:
                raise ValueError("end_line must be greater than start_line")

        return v

    @validator("target_file")
    def validate_target_file(cls, v):
        """Validate target file exists and is a test file."""
        if not v:
            raise ValueError("Target file path cannot be empty")

        file_path = Path(v)
        if not file_path.exists():
            raise ValueError(f"Target file does not exist: {v}")

        if not file_path.suffix in [".js", ".ts"]:
            raise ValueError(f"Target file must be JavaScript or TypeScript: {v}")

        return str(file_path)

    @property
    def is_applied(self) -> bool:
        """Check if patch has been applied."""
        return self.applied_at is not None

    @property
    def affects_multiple_lines(self) -> bool:
        """Check if patch affects multiple lines."""
        return (
            self.operation
            in [PatchOperation.REPLACE_BLOCK, PatchOperation.DELETE_BLOCK]
            and self.end_line is not None
        )

    def get_affected_line_range(self) -> Tuple[int, int]:
        """Get the range of lines affected by this patch."""
        if self.affects_multiple_lines:
            return (self.start_line, self.end_line)
        else:
            return (self.start_line, self.start_line)


class PatchResult(BaseModel):
    """Result of applying a code patch."""

    model_config = ConfigDict(extra="forbid")

    # Patch information
    patch_id: str = Field(..., description="ID of the applied patch")
    target_file: str = Field(..., description="File that was patched")

    # Application result
    success: bool = Field(..., description="Whether patch was applied successfully")
    applied_at: datetime = Field(
        default_factory=datetime.utcnow, description="When patch was applied"
    )

    # File changes
    backup_created: bool = Field(..., description="Whether backup was created")
    backup_path: Optional[str] = Field(None, description="Path to backup file")
    lines_changed: int = Field(..., ge=0, description="Number of lines changed")

    # Validation results
    validation_result: Optional[PatchValidationResult] = Field(
        None, description="Validation result after applying patch"
    )

    # Error information
    error_message: Optional[str] = Field(
        None, description="Error message if patch failed"
    )
    rollback_available: bool = Field(..., description="Whether rollback is possible")

    # Content tracking
    original_content_hash: str = Field(..., description="Hash of original content")
    patched_content_hash: str = Field(..., description="Hash of patched content")

    @property
    def needs_rollback(self) -> bool:
        """Check if patch needs to be rolled back due to validation failures."""
        return not self.success or (
            self.validation_result and self.validation_result.has_errors
        )

    def to_summary(self) -> Dict[str, Any]:
        """Create a summary dictionary for logging."""
        return {
            "patch_id": self.patch_id,
            "target_file": self.target_file,
            "success": self.success,
            "lines_changed": self.lines_changed,
            "backup_created": self.backup_created,
            "validation_passed": (
                self.validation_result.is_valid if self.validation_result else None
            ),
            "needs_rollback": self.needs_rollback,
            "applied_at": self.applied_at.isoformat(),
        }


class FlakyTestDetection(BaseModel):
    """Detection result for flaky test patterns."""

    model_config = ConfigDict(extra="forbid")

    test_name: str = Field(..., description="Name of the test")
    test_file: str = Field(..., description="Path to test file")

    # Flakiness indicators
    is_flaky: bool = Field(..., description="Whether test is detected as flaky")
    flakiness_score: float = Field(
        ..., ge=0.0, le=1.0, description="Flakiness score (0-1)"
    )
    confidence: ConfidenceLevel = Field(
        ..., description="Confidence in flaky detection"
    )

    # Flakiness patterns
    detected_patterns: List[str] = Field(
        default_factory=list, description="Detected flaky patterns"
    )
    failure_frequency: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Historical failure frequency"
    )

    # Stability improvements
    stability_suggestions: List[FixSuggestion] = Field(
        default_factory=list, description="Suggestions to improve test stability"
    )

    # Analysis metadata
    analysis_period: Optional[str] = Field(
        None, description="Time period analyzed for flakiness"
    )
    sample_size: Optional[int] = Field(
        None, ge=0, description="Number of test runs analyzed"
    )

    @property
    def needs_stability_improvements(self) -> bool:
        """Check if test needs stability improvements."""
        return self.is_flaky and len(self.stability_suggestions) > 0

    @property
    def risk_level(self) -> str:
        """Get risk level based on flakiness score."""
        if self.flakiness_score >= 0.7:
            return "high"
        elif self.flakiness_score >= 0.4:
            return "medium"
        else:
            return "low"


class CodePatcherError(Exception):
    """Exception raised during code patching operations."""

    def __init__(
        self,
        message: str,
        patch_id: Optional[str] = None,
        target_file: Optional[str] = None,
        operation: Optional[PatchOperation] = None,
    ):
        super().__init__(message)
        self.patch_id = patch_id
        self.target_file = target_file
        self.operation = operation

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            "error": str(self),
            "patch_id": self.patch_id,
            "target_file": self.target_file,
            "operation": self.operation.value if self.operation else None,
        }
