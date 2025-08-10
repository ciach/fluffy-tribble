"""
Unit tests for the code patcher.

Tests patch application, validation, backup creation, rollback,
and flaky test detection.
"""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, mock_open

from orchestrator.analysis.patcher import CodePatcher
from orchestrator.analysis.models import (
    CodePatch,
    PatchResult,
    PatchValidationResult,
    FlakyTestDetection,
    FixSuggestion,
    PatchOperation,
    ConfidenceLevel,
    FixType,
    CodePatcherError,
)
from orchestrator.core.config import Config
from orchestrator.core.exceptions import ValidationError
from orchestrator.mcp.filesystem_client import FilesystemMCPClient


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = Mock(spec=Config)
    config.debug_enabled = False
    return config


@pytest.fixture
def mock_filesystem_client():
    """Create a mock filesystem client."""
    client = Mock(spec=FilesystemMCPClient)
    return client


@pytest.fixture
def code_patcher(mock_config, mock_filesystem_client):
    """Create a code patcher instance."""
    with patch("pathlib.Path.mkdir"):
        return CodePatcher(mock_config, mock_filesystem_client)


@pytest.fixture
def sample_test_content():
    """Sample test file content."""
    return """import { test, expect } from '@playwright/test';

test('sample test', async ({ page }) => {
  await page.goto('https://example.com');
  await page.getByRole('button', { name: 'Submit' }).click();
  await expect(page.getByText('Success')).toBeVisible();
});"""


@pytest.fixture
def sample_fix_suggestion():
    """Create a sample fix suggestion."""
    return FixSuggestion(
        fix_type=FixType.WAIT_CONDITION,
        description="Add explicit wait for button",
        confidence=ConfidenceLevel.HIGH,
        original_code="await page.getByRole('button', { name: 'Submit' }).click();",
        suggested_code="await expect(page.getByRole('button', { name: 'Submit' })).toBeVisible();\n  await page.getByRole('button', { name: 'Submit' }).click();",
        line_number=4,
        reasoning="Explicit wait ensures element is ready",
        potential_side_effects=[],
        test_impact="minimal",
    )


@pytest.fixture
def sample_patch(sample_test_content):
    """Create a sample code patch."""
    with patch("pathlib.Path.exists", return_value=True):
        return CodePatch(
            patch_id="test-patch-123",
            target_file="e2e/test.spec.ts",
            original_content=sample_test_content,
            operation=PatchOperation.REPLACE_LINE,
            start_line=4,
            original_code="await page.getByRole('button', { name: 'Submit' }).click();",
            patched_code="await expect(page.getByRole('button', { name: 'Submit' })).toBeVisible();\n  await page.getByRole('button', { name: 'Submit' }).click();",
            description="Add explicit wait for button",
            reasoning="Explicit wait ensures element is ready",
            confidence=ConfidenceLevel.HIGH,
        )


class TestCodePatcher:
    """Test cases for CodePatcher."""

    def test_init(self, mock_config, mock_filesystem_client):
        """Test patcher initialization."""
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            patcher = CodePatcher(mock_config, mock_filesystem_client)

            assert patcher.config == mock_config
            assert patcher.filesystem_client == mock_filesystem_client
            assert patcher.backup_dir == Path("e2e/.backups")
            assert patcher.max_backup_age_days == 7
            assert patcher.validation_timeout == 30

            # Verify backup directory creation
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @pytest.mark.asyncio
    async def test_apply_patch_success(
        self, code_patcher, sample_patch, sample_test_content, mock_filesystem_client
    ):
        """Test successful patch application."""
        # Mock file operations
        mock_filesystem_client.read_file = AsyncMock(return_value=sample_test_content)
        mock_filesystem_client.write_file = AsyncMock()

        # Mock validation methods
        with patch.object(
            code_patcher, "_validate_patch_preconditions", new_callable=AsyncMock
        ), patch.object(
            code_patcher,
            "_create_backup",
            new_callable=AsyncMock,
            return_value="/backup/path",
        ), patch.object(
            code_patcher, "_validate_patched_file", new_callable=AsyncMock
        ) as mock_validate:

            mock_validate.return_value = PatchValidationResult(
                is_valid=True,
                validation_errors=[],
                warnings=[],
                syntax_valid=True,
                semantic_changes=[],
            )

            result = await code_patcher.apply_patch(sample_patch)

            # Verify result
            assert isinstance(result, PatchResult)
            assert result.patch_id == "test-patch-123"
            assert result.success is True
            assert result.backup_created is True
            assert result.backup_path == "/backup/path"
            assert result.lines_changed > 0
            assert result.validation_result.is_valid is True

            # Verify file was written
            mock_filesystem_client.write_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_apply_patch_dry_run(
        self, code_patcher, sample_patch, sample_test_content, mock_filesystem_client
    ):
        """Test patch application in dry run mode."""
        # Mock file operations
        mock_filesystem_client.read_file = AsyncMock(return_value=sample_test_content)
        mock_filesystem_client.write_file = AsyncMock()

        # Mock validation methods
        with patch.object(
            code_patcher, "_validate_patch_preconditions", new_callable=AsyncMock
        ), patch.object(
            code_patcher, "_validate_patched_file", new_callable=AsyncMock
        ) as mock_validate:

            mock_validate.return_value = PatchValidationResult(
                is_valid=True,
                validation_errors=[],
                warnings=[],
                syntax_valid=True,
                semantic_changes=[],
            )

            result = await code_patcher.apply_patch(sample_patch, dry_run=True)

            # Verify result
            assert result.success is True
            assert result.backup_created is False  # No backup in dry run
            assert result.backup_path is None

            # Verify file was NOT written
            mock_filesystem_client.write_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_apply_patch_validation_failure(
        self, code_patcher, sample_patch, sample_test_content, mock_filesystem_client
    ):
        """Test patch application with validation failure."""
        # Mock file operations
        mock_filesystem_client.read_file = AsyncMock(return_value=sample_test_content)
        mock_filesystem_client.write_file = AsyncMock()

        # Mock validation methods
        with patch.object(
            code_patcher, "_validate_patch_preconditions", new_callable=AsyncMock
        ), patch.object(
            code_patcher,
            "_create_backup",
            new_callable=AsyncMock,
            return_value="/backup/path",
        ), patch.object(
            code_patcher, "_validate_patched_file", new_callable=AsyncMock
        ) as mock_validate, patch.object(
            code_patcher, "_rollback_patch", new_callable=AsyncMock
        ) as mock_rollback:

            mock_validate.return_value = PatchValidationResult(
                is_valid=False,
                validation_errors=["Syntax error"],
                warnings=[],
                syntax_valid=False,
                semantic_changes=[],
            )

            result = await code_patcher.apply_patch(sample_patch)

            # Verify rollback was called
            mock_rollback.assert_called_once()
            assert result.needs_rollback is True

    @pytest.mark.asyncio
    async def test_create_patch_from_suggestion(
        self,
        code_patcher,
        sample_fix_suggestion,
        sample_test_content,
        mock_filesystem_client,
    ):
        """Test creating patch from fix suggestion."""
        # Mock file operations
        mock_filesystem_client.read_file = AsyncMock(return_value=sample_test_content)

        patch = await code_patcher.create_patch_from_suggestion(
            sample_fix_suggestion, "e2e/test.spec.ts"
        )

        # Verify patch
        assert isinstance(patch, CodePatch)
        assert patch.target_file == "e2e/test.spec.ts"
        assert patch.operation == PatchOperation.REPLACE_LINE
        assert patch.start_line == 4
        assert patch.confidence == ConfidenceLevel.HIGH
        assert patch.requires_backup is True
        assert patch.requires_validation is True

    @pytest.mark.asyncio
    async def test_detect_flaky_test(self, code_patcher, mock_filesystem_client):
        """Test flaky test detection."""
        # Sample flaky test content
        flaky_content = """
        test('flaky test', async ({ page }) => {
          await page.waitForTimeout(5000);  // Hard-coded wait
          await Promise.all([
            page.goto('https://example.com'),
            page.click('button')
          ]);  // Race condition
          const response = await fetch('/api/data');  // Network dependency
        });
        """

        mock_filesystem_client.read_file = AsyncMock(return_value=flaky_content)

        # Mock failure history
        failure_history = [
            {"success": False, "timestamp": "2024-01-01"},
            {"success": True, "timestamp": "2024-01-02"},
            {"success": False, "timestamp": "2024-01-03"},
        ]

        detection = await code_patcher.detect_flaky_test(
            "e2e/flaky.spec.ts", "flaky test", failure_history
        )

        # Verify detection
        assert isinstance(detection, FlakyTestDetection)
        assert detection.test_name == "flaky test"
        assert detection.is_flaky is True
        assert detection.flakiness_score > 0.3
        assert "hard_coded_waits" in detection.detected_patterns
        assert "potential_race_conditions" in detection.detected_patterns
        assert "network_dependencies" in detection.detected_patterns
        assert len(detection.stability_suggestions) > 0
        assert detection.failure_frequency == 2 / 3  # 2 failures out of 3 runs

    @pytest.mark.asyncio
    async def test_rollback_patch(self, code_patcher):
        """Test patch rollback."""
        # Create temporary files for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            target_file = Path(temp_dir) / "test.spec.ts"
            backup_file = Path(temp_dir) / "backup.spec.ts"

            # Create files
            target_file.write_text("modified content")
            backup_file.write_text("original content")

            # Create patch result
            patch_result = PatchResult(
                patch_id="test-patch",
                target_file=str(target_file),
                success=True,
                backup_created=True,
                backup_path=str(backup_file),
                lines_changed=1,
                rollback_available=True,
                original_content_hash="hash1",
                patched_content_hash="hash2",
            )

            # Rollback
            success = await code_patcher.rollback_patch(patch_result)

            # Verify rollback
            assert success is True
            assert target_file.read_text() == "original content"

    @pytest.mark.asyncio
    async def test_rollback_patch_no_backup(self, code_patcher):
        """Test rollback when no backup is available."""
        patch_result = PatchResult(
            patch_id="test-patch",
            target_file="test.spec.ts",
            success=True,
            backup_created=False,
            backup_path=None,
            lines_changed=1,
            rollback_available=False,
            original_content_hash="hash1",
            patched_content_hash="hash2",
        )

        with pytest.raises(CodePatcherError) as exc_info:
            await code_patcher.rollback_patch(patch_result)

        assert "no backup available" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_validate_patch_preconditions_file_not_exists(
        self, code_patcher, sample_patch
    ):
        """Test validation when target file doesn't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(ValidationError) as exc_info:
                await code_patcher._validate_patch_preconditions(sample_patch)

            assert "Target file does not exist" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_validate_patch_preconditions_outside_e2e(
        self, code_patcher, sample_patch
    ):
        """Test validation when file is outside e2e directory."""
        sample_patch.target_file = "/etc/passwd"  # Outside e2e

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.resolve"
        ) as mock_resolve:

            # Mock resolve to return paths outside e2e
            mock_resolve.side_effect = lambda: Path("/etc/passwd")

            with pytest.raises(ValidationError) as exc_info:
                await code_patcher._validate_patch_preconditions(sample_patch)

            assert "must be in e2e directory" in str(exc_info.value)

    def test_apply_patch_operation_replace_line(self, code_patcher, sample_patch):
        """Test replace line patch operation."""
        original_content = """line 1
line 2
line 3
line 4
line 5"""

        sample_patch.operation = PatchOperation.REPLACE_LINE
        sample_patch.start_line = 3
        sample_patch.patched_code = "new line 3"

        result = code_patcher._apply_patch_operation(sample_patch, original_content)

        lines = result.splitlines()
        assert lines[2] == "new line 3"
        assert lines[1] == "line 2"  # Other lines unchanged
        assert lines[3] == "line 4"

    def test_apply_patch_operation_replace_block(self, code_patcher, sample_patch):
        """Test replace block patch operation."""
        original_content = """line 1
line 2
line 3
line 4
line 5"""

        sample_patch.operation = PatchOperation.REPLACE_BLOCK
        sample_patch.start_line = 2
        sample_patch.end_line = 4
        sample_patch.patched_code = "new block line 1\nnew block line 2"

        result = code_patcher._apply_patch_operation(sample_patch, original_content)

        lines = result.splitlines()
        assert lines[0] == "line 1"
        assert lines[1] == "new block line 1"
        assert lines[2] == "new block line 2"
        assert lines[3] == "line 5"

    def test_apply_patch_operation_insert_before(self, code_patcher, sample_patch):
        """Test insert before patch operation."""
        original_content = """line 1
line 2
line 3"""

        sample_patch.operation = PatchOperation.INSERT_BEFORE
        sample_patch.start_line = 2
        sample_patch.patched_code = "inserted line"

        result = code_patcher._apply_patch_operation(sample_patch, original_content)

        lines = result.splitlines()
        assert lines[0] == "line 1"
        assert lines[1] == "inserted line"
        assert lines[2] == "line 2"
        assert lines[3] == "line 3"

    def test_apply_patch_operation_delete_line(self, code_patcher, sample_patch):
        """Test delete line patch operation."""
        original_content = """line 1
line 2
line 3
line 4"""

        sample_patch.operation = PatchOperation.DELETE_LINE
        sample_patch.start_line = 2

        result = code_patcher._apply_patch_operation(sample_patch, original_content)

        lines = result.splitlines()
        assert len(lines) == 3
        assert lines[0] == "line 1"
        assert lines[1] == "line 3"
        assert lines[2] == "line 4"

    def test_find_code_line(self, code_patcher):
        """Test finding line number for target code."""
        content = """import { test } from '@playwright/test';

test('sample', async ({ page }) => {
  await page.goto('https://example.com');
  await page.click('button');
});"""

        line_num = code_patcher._find_code_line(content, "page.click('button')")
        assert line_num == 5

    def test_generate_wait_condition_fix(self, code_patcher):
        """Test generating wait condition fix."""
        original_code = "await page.getByRole('button').click();"

        result = code_patcher._generate_wait_condition_fix(original_code)

        assert "expect(" in result
        assert "toBeVisible()" in result
        assert original_code in result

    def test_generate_selector_replacement_fix(self, code_patcher):
        """Test generating selector replacement fix."""
        original_code = "await page.locator('#submit-btn').click();"

        result = code_patcher._generate_selector_replacement_fix(original_code)

        assert "getByTestId(" in result
        assert "#" not in result

    def test_generate_timeout_adjustment_fix(self, code_patcher):
        """Test generating timeout adjustment fix."""
        original_code = "await page.click('button');"

        result = code_patcher._generate_timeout_adjustment_fix(original_code)

        assert "timeout: 10000" in result

    def test_generate_error_handling_fix(self, code_patcher):
        """Test generating error handling fix."""
        original_code = "await page.click('button');"

        result = code_patcher._generate_error_handling_fix(original_code)

        assert "try {" in result
        assert "catch (error)" in result
        assert original_code in result

    def test_analyze_flaky_patterns(self, code_patcher):
        """Test analyzing flaky patterns in test content."""
        content = """
        test('flaky test', async ({ page }) => {
          await page.waitForTimeout(5000);
          await Promise.all([page.goto('/'), page.click('button')]);
          const response = await fetch('/api/data');
          const date = new Date();
          await fs.readFile('test.txt');
        });
        """

        patterns = code_patcher._analyze_flaky_patterns(content, "flaky test")

        assert "hard_coded_waits" in patterns
        assert "potential_race_conditions" in patterns
        assert "network_dependencies" in patterns
        assert "time_based_logic" in patterns
        assert "file_system_dependencies" in patterns

    def test_calculate_flakiness_score(self, code_patcher):
        """Test calculating flakiness score."""
        patterns = ["hard_coded_waits", "potential_race_conditions"]
        failure_history = [
            {"success": False},
            {"success": True},
            {"success": False},
            {"success": True},
        ]

        score = code_patcher._calculate_flakiness_score(patterns, failure_history)

        # Should be > 0 due to patterns and failure rate
        assert score > 0.5
        assert score <= 1.0

    def test_check_playwright_best_practices(self, code_patcher):
        """Test checking Playwright best practices."""
        content = """
        await page.waitForTimeout(5000);
        await page.locator('.submit-button').click();
        await page.locator('#username').fill('test');
        await page.goto('/login');
        """

        warnings = code_patcher._check_playwright_best_practices(content)

        assert any("Hard-coded timeout" in warning for warning in warnings)
        assert any("CSS selector found" in warning for warning in warnings)
        assert any("missing error handling" in warning for warning in warnings)

    def test_calculate_lines_changed(self, code_patcher):
        """Test calculating number of lines changed."""
        original = """line 1
line 2
line 3"""

        patched = """line 1
modified line 2
line 3
new line 4"""

        changed = code_patcher._calculate_lines_changed(original, patched)

        # Should detect 2 changes: modified line 2 and new line 4
        assert changed == 2


class TestCodePatch:
    """Test cases for CodePatch model."""

    def test_valid_code_patch(self, sample_test_content):
        """Test creating valid code patch."""
        with patch("pathlib.Path.exists", return_value=True):
            patch = CodePatch(
                patch_id="test-123",
                target_file="e2e/test.spec.ts",
                original_content=sample_test_content,
                operation=PatchOperation.REPLACE_LINE,
                start_line=4,
                original_code="old code",
                patched_code="new code",
                description="Test patch",
                reasoning="For testing",
                confidence=ConfidenceLevel.HIGH,
            )

            assert patch.patch_id == "test-123"
            assert patch.operation == PatchOperation.REPLACE_LINE
            assert patch.start_line == 4
            assert not patch.is_applied
            assert not patch.affects_multiple_lines

    def test_block_operation_validation(self, sample_test_content):
        """Test validation for block operations."""
        with patch("pathlib.Path.exists", return_value=True):
            # Should require end_line for block operations
            with pytest.raises(ValueError) as exc_info:
                CodePatch(
                    patch_id="test-123",
                    target_file="e2e/test.spec.ts",
                    original_content=sample_test_content,
                    operation=PatchOperation.REPLACE_BLOCK,
                    start_line=4,
                    end_line=None,  # Missing end_line
                    original_code="old code",
                    patched_code="new code",
                    description="Test patch",
                    reasoning="For testing",
                    confidence=ConfidenceLevel.HIGH,
                )

            assert "end_line is required" in str(exc_info.value)

    def test_invalid_line_range(self, sample_test_content):
        """Test validation of invalid line range."""
        with patch("pathlib.Path.exists", return_value=True):
            with pytest.raises(ValueError) as exc_info:
                CodePatch(
                    patch_id="test-123",
                    target_file="e2e/test.spec.ts",
                    original_content=sample_test_content,
                    operation=PatchOperation.REPLACE_BLOCK,
                    start_line=5,
                    end_line=3,  # end_line < start_line
                    original_code="old code",
                    patched_code="new code",
                    description="Test patch",
                    reasoning="For testing",
                    confidence=ConfidenceLevel.HIGH,
                )

            assert "end_line must be greater than start_line" in str(exc_info.value)

    def test_get_affected_line_range(self, sample_test_content):
        """Test getting affected line range."""
        with patch("pathlib.Path.exists", return_value=True):
            # Single line patch
            single_patch = CodePatch(
                patch_id="test-123",
                target_file="e2e/test.spec.ts",
                original_content=sample_test_content,
                operation=PatchOperation.REPLACE_LINE,
                start_line=4,
                original_code="old code",
                patched_code="new code",
                description="Test patch",
                reasoning="For testing",
                confidence=ConfidenceLevel.HIGH,
            )

            start, end = single_patch.get_affected_line_range()
            assert start == 4
            assert end == 4

            # Block patch
            block_patch = CodePatch(
                patch_id="test-456",
                target_file="e2e/test.spec.ts",
                original_content=sample_test_content,
                operation=PatchOperation.REPLACE_BLOCK,
                start_line=2,
                end_line=5,
                original_code="old code",
                patched_code="new code",
                description="Test patch",
                reasoning="For testing",
                confidence=ConfidenceLevel.HIGH,
            )

            start, end = block_patch.get_affected_line_range()
            assert start == 2
            assert end == 5


class TestPatchResult:
    """Test cases for PatchResult model."""

    def test_valid_patch_result(self):
        """Test creating valid patch result."""
        result = PatchResult(
            patch_id="test-123",
            target_file="test.spec.ts",
            success=True,
            backup_created=True,
            backup_path="/backup/path",
            lines_changed=3,
            rollback_available=True,
            original_content_hash="hash1",
            patched_content_hash="hash2",
        )

        assert result.patch_id == "test-123"
        assert result.success is True
        assert result.backup_created is True
        assert result.lines_changed == 3
        assert not result.needs_rollback  # Success with no validation errors

    def test_needs_rollback_validation_errors(self):
        """Test needs_rollback with validation errors."""
        validation_result = PatchValidationResult(
            is_valid=False,
            validation_errors=["Syntax error"],
            warnings=[],
            syntax_valid=False,
            semantic_changes=[],
        )

        result = PatchResult(
            patch_id="test-123",
            target_file="test.spec.ts",
            success=True,
            backup_created=True,
            lines_changed=1,
            validation_result=validation_result,
            rollback_available=True,
            original_content_hash="hash1",
            patched_content_hash="hash2",
        )

        assert result.needs_rollback is True

    def test_to_summary(self):
        """Test creating summary dictionary."""
        result = PatchResult(
            patch_id="test-123",
            target_file="test.spec.ts",
            success=True,
            backup_created=True,
            lines_changed=2,
            rollback_available=True,
            original_content_hash="hash1",
            patched_content_hash="hash2",
        )

        summary = result.to_summary()

        assert summary["patch_id"] == "test-123"
        assert summary["success"] is True
        assert summary["lines_changed"] == 2
        assert summary["backup_created"] is True
        assert "applied_at" in summary


class TestFlakyTestDetection:
    """Test cases for FlakyTestDetection model."""

    def test_valid_flaky_detection(self):
        """Test creating valid flaky test detection."""
        stability_suggestions = [
            FixSuggestion(
                fix_type=FixType.WAIT_CONDITION,
                description="Add wait",
                confidence=ConfidenceLevel.HIGH,
                reasoning="Will help",
            )
        ]

        detection = FlakyTestDetection(
            test_name="flaky_test",
            test_file="test.spec.ts",
            is_flaky=True,
            flakiness_score=0.7,
            confidence=ConfidenceLevel.HIGH,
            detected_patterns=["hard_coded_waits"],
            failure_frequency=0.3,
            stability_suggestions=stability_suggestions,
            sample_size=10,
        )

        assert detection.test_name == "flaky_test"
        assert detection.is_flaky is True
        assert detection.flakiness_score == 0.7
        assert detection.needs_stability_improvements is True
        assert detection.risk_level == "high"

    def test_risk_level_calculation(self):
        """Test risk level calculation based on flakiness score."""
        # High risk
        high_risk = FlakyTestDetection(
            test_name="test",
            test_file="test.spec.ts",
            is_flaky=True,
            flakiness_score=0.8,
            confidence=ConfidenceLevel.HIGH,
        )
        assert high_risk.risk_level == "high"

        # Medium risk
        medium_risk = FlakyTestDetection(
            test_name="test",
            test_file="test.spec.ts",
            is_flaky=True,
            flakiness_score=0.5,
            confidence=ConfidenceLevel.MEDIUM,
        )
        assert medium_risk.risk_level == "medium"

        # Low risk
        low_risk = FlakyTestDetection(
            test_name="test",
            test_file="test.spec.ts",
            is_flaky=False,
            flakiness_score=0.2,
            confidence=ConfidenceLevel.LOW,
        )
        assert low_risk.risk_level == "low"
