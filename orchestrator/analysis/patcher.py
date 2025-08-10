"""
Code patcher with validation for test files.

Applies targeted fixes to resolve test issues with backup creation,
validation, and rollback capabilities.
"""

import hashlib
import logging
import re
import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from ..core.config import Config
from ..core.exceptions import ValidationError
from ..mcp.filesystem_client import FilesystemMCPClient
from .models import (
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

logger = logging.getLogger(__name__)


class CodePatcher:
    """
    Applies targeted code patches to test files with validation and backup.

    Provides safe patching operations with automatic backup creation,
    syntax validation, and rollback capabilities.
    """

    def __init__(self, config: Config, filesystem_client: FilesystemMCPClient):
        """Initialize the code patcher."""
        self.config = config
        self.filesystem_client = filesystem_client

        # Patcher configuration
        self.backup_dir = Path("e2e/.backups")
        self.max_backup_age_days = 7
        self.validation_timeout = 30  # seconds

        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Code patcher initialized")

    async def apply_patch(self, patch: CodePatch, dry_run: bool = False) -> PatchResult:
        """
        Apply a code patch to a test file.

        Args:
            patch: The patch to apply
            dry_run: If True, validate patch without applying

        Returns:
            PatchResult with application details

        Raises:
            CodePatcherError: If patch application fails
            ValidationError: If patch validation fails
        """
        logger.info(
            f"Applying patch {patch.patch_id} to {patch.target_file} "
            f"(dry_run: {dry_run})"
        )

        try:
            # Validate patch before applying
            await self._validate_patch_preconditions(patch)

            # Create backup if not dry run
            backup_path = None
            if not dry_run and patch.requires_backup:
                backup_path = await self._create_backup(patch.target_file)

            # Calculate original content hash
            original_content = await self._read_file_content(patch.target_file)
            original_hash = self._calculate_content_hash(original_content)

            # Apply the patch
            patched_content = self._apply_patch_operation(patch, original_content)

            # Calculate patched content hash
            patched_hash = self._calculate_content_hash(patched_content)

            # Write patched content if not dry run
            if not dry_run:
                await self._write_file_content(patch.target_file, patched_content)

            # Validate the patched file
            validation_result = None
            if patch.requires_validation:
                validation_result = await self._validate_patched_file(
                    patch.target_file, patched_content, dry_run
                )

            # Calculate lines changed
            lines_changed = self._calculate_lines_changed(
                original_content, patched_content
            )

            # Create result
            result = PatchResult(
                patch_id=patch.patch_id,
                target_file=patch.target_file,
                success=True,
                backup_created=backup_path is not None,
                backup_path=backup_path,
                lines_changed=lines_changed,
                validation_result=validation_result,
                rollback_available=backup_path is not None,
                original_content_hash=original_hash,
                patched_content_hash=patched_hash,
            )

            # Check if rollback is needed
            if result.needs_rollback and not dry_run:
                logger.warning(
                    f"Patch {patch.patch_id} needs rollback due to validation failures"
                )
                await self._rollback_patch(result)

            logger.info(
                f"Patch {patch.patch_id} applied successfully "
                f"({lines_changed} lines changed)"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to apply patch {patch.patch_id}: {e}")
            raise CodePatcherError(
                f"Patch application failed: {e}",
                patch_id=patch.patch_id,
                target_file=patch.target_file,
                operation=patch.operation,
            )

    async def create_patch_from_suggestion(
        self,
        fix_suggestion: FixSuggestion,
        target_file: str,
        analysis_context: Optional[Dict[str, Any]] = None,
    ) -> CodePatch:
        """
        Create a code patch from a fix suggestion.

        Args:
            fix_suggestion: The fix suggestion to implement
            target_file: Path to the target file
            analysis_context: Optional context from failure analysis

        Returns:
            CodePatch ready to be applied

        Raises:
            ValidationError: If patch creation fails
        """
        logger.info(
            f"Creating patch from {fix_suggestion.fix_type.value} suggestion "
            f"for {target_file}"
        )

        try:
            # Read original file content
            original_content = await self._read_file_content(target_file)

            # Determine patch operation based on fix type
            operation, start_line, end_line = await self._determine_patch_operation(
                fix_suggestion, original_content
            )

            # Extract original code to be replaced
            original_code = self._extract_original_code(
                original_content, start_line, end_line, operation
            )

            # Generate patched code
            patched_code = await self._generate_patched_code(
                fix_suggestion, original_code, analysis_context
            )

            # Create patch
            patch = CodePatch(
                patch_id=str(uuid.uuid4()),
                fix_suggestion_id=getattr(fix_suggestion, "id", None),
                target_file=target_file,
                original_content=original_content,
                operation=operation,
                start_line=start_line,
                end_line=end_line,
                original_code=original_code,
                patched_code=patched_code,
                description=fix_suggestion.description,
                reasoning=fix_suggestion.reasoning,
                confidence=fix_suggestion.confidence,
                requires_backup=True,
                requires_validation=True,
            )

            logger.info(
                f"Created patch {patch.patch_id} for {fix_suggestion.fix_type.value}"
            )
            return patch

        except Exception as e:
            logger.error(f"Failed to create patch from suggestion: {e}")
            raise ValidationError(
                f"Patch creation failed: {e}", validation_type="patch_creation"
            )

    async def detect_flaky_test(
        self,
        test_file: str,
        test_name: str,
        failure_history: Optional[List[Dict[str, Any]]] = None,
    ) -> FlakyTestDetection:
        """
        Detect if a test is flaky and suggest stability improvements.

        Args:
            test_file: Path to the test file
            test_name: Name of the test
            failure_history: Optional historical failure data

        Returns:
            FlakyTestDetection with analysis results
        """
        logger.info(f"Analyzing flakiness for {test_name} in {test_file}")

        try:
            # Read test file content
            test_content = await self._read_file_content(test_file)

            # Analyze test patterns for flakiness indicators
            flaky_patterns = await self._analyze_flaky_patterns(test_content, test_name)

            # Calculate flakiness score
            flakiness_score = self._calculate_flakiness_score(
                flaky_patterns, failure_history
            )

            # Determine if test is flaky
            is_flaky = flakiness_score >= 0.3  # Threshold for flaky detection

            # Generate stability suggestions
            stability_suggestions = await self._generate_stability_suggestions(
                test_content, flaky_patterns, test_name
            )

            # Calculate failure frequency if history available
            failure_frequency = None
            if failure_history:
                total_runs = len(failure_history)
                failures = sum(
                    1 for run in failure_history if not run.get("success", True)
                )
                failure_frequency = failures / total_runs if total_runs > 0 else 0.0

            detection = FlakyTestDetection(
                test_name=test_name,
                test_file=test_file,
                is_flaky=is_flaky,
                flakiness_score=flakiness_score,
                confidence=ConfidenceLevel.MEDIUM,  # Conservative confidence
                detected_patterns=flaky_patterns,
                failure_frequency=failure_frequency,
                stability_suggestions=stability_suggestions,
                sample_size=len(failure_history) if failure_history else None,
            )

            logger.info(
                f"Flaky test analysis complete: {test_name} "
                f"(flaky: {is_flaky}, score: {flakiness_score:.2f})"
            )

            return detection

        except Exception as e:
            logger.error(f"Failed to detect flaky test: {e}")
            raise CodePatcherError(
                f"Flaky test detection failed: {e}", target_file=test_file
            )

    async def rollback_patch(self, patch_result: PatchResult) -> bool:
        """
        Rollback a previously applied patch.

        Args:
            patch_result: Result of the patch to rollback

        Returns:
            True if rollback was successful

        Raises:
            CodePatcherError: If rollback fails
        """
        if not patch_result.rollback_available or not patch_result.backup_path:
            raise CodePatcherError(
                "Cannot rollback patch - no backup available",
                patch_id=patch_result.patch_id,
                target_file=patch_result.target_file,
            )

        logger.info(f"Rolling back patch {patch_result.patch_id}")

        try:
            # Restore from backup
            backup_path = Path(patch_result.backup_path)
            target_path = Path(patch_result.target_file)

            if not backup_path.exists():
                raise CodePatcherError(
                    f"Backup file not found: {backup_path}",
                    patch_id=patch_result.patch_id,
                )

            # Copy backup back to original location
            shutil.copy2(backup_path, target_path)

            logger.info(f"Successfully rolled back patch {patch_result.patch_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to rollback patch {patch_result.patch_id}: {e}")
            raise CodePatcherError(
                f"Rollback failed: {e}",
                patch_id=patch_result.patch_id,
                target_file=patch_result.target_file,
            )

    async def _validate_patch_preconditions(self, patch: CodePatch) -> None:
        """Validate patch preconditions before applying."""
        # Check target file exists
        target_path = Path(patch.target_file)
        if not target_path.exists():
            raise ValidationError(
                f"Target file does not exist: {patch.target_file}",
                validation_type="file_existence",
            )

        # Check file is in e2e directory (safety restriction)
        if not str(target_path.resolve()).startswith(str(Path("e2e").resolve())):
            raise ValidationError(
                f"Target file must be in e2e directory: {patch.target_file}",
                validation_type="file_location",
            )

        # Validate line numbers
        content = await self._read_file_content(patch.target_file)
        total_lines = len(content.splitlines())

        if patch.start_line > total_lines:
            raise ValidationError(
                f"Start line {patch.start_line} exceeds file length {total_lines}",
                validation_type="line_range",
            )

        if patch.end_line and patch.end_line > total_lines:
            raise ValidationError(
                f"End line {patch.end_line} exceeds file length {total_lines}",
                validation_type="line_range",
            )

    async def _create_backup(self, file_path: str) -> str:
        """Create a backup of the file before patching."""
        source_path = Path(file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{source_path.stem}_{timestamp}{source_path.suffix}"
        backup_path = self.backup_dir / backup_name

        # Copy file to backup location
        shutil.copy2(source_path, backup_path)

        logger.debug(f"Created backup: {backup_path}")
        return str(backup_path)

    async def _read_file_content(self, file_path: str) -> str:
        """Read file content using filesystem client."""
        try:
            return await self.filesystem_client.read_file(file_path)
        except Exception as e:
            raise CodePatcherError(f"Failed to read file {file_path}: {e}")

    async def _write_file_content(self, file_path: str, content: str) -> None:
        """Write file content using filesystem client."""
        try:
            await self.filesystem_client.write_file(file_path, content)
        except Exception as e:
            raise CodePatcherError(f"Failed to write file {file_path}: {e}")

    def _calculate_content_hash(self, content: str) -> str:
        """Calculate SHA-256 hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _apply_patch_operation(self, patch: CodePatch, original_content: str) -> str:
        """Apply the patch operation to the content."""
        lines = original_content.splitlines(keepends=True)

        if patch.operation == PatchOperation.REPLACE_LINE:
            # Replace single line
            if patch.start_line <= len(lines):
                lines[patch.start_line - 1] = patch.patched_code + "\n"

        elif patch.operation == PatchOperation.REPLACE_BLOCK:
            # Replace block of lines
            start_idx = patch.start_line - 1
            end_idx = patch.end_line if patch.end_line else patch.start_line

            # Replace the block
            new_lines = patch.patched_code.splitlines(keepends=True)
            lines[start_idx:end_idx] = new_lines

        elif patch.operation == PatchOperation.INSERT_BEFORE:
            # Insert before specified line
            insert_idx = patch.start_line - 1
            new_lines = patch.patched_code.splitlines(keepends=True)
            lines[insert_idx:insert_idx] = new_lines

        elif patch.operation == PatchOperation.INSERT_AFTER:
            # Insert after specified line
            insert_idx = patch.start_line
            new_lines = patch.patched_code.splitlines(keepends=True)
            lines[insert_idx:insert_idx] = new_lines

        elif patch.operation == PatchOperation.DELETE_LINE:
            # Delete single line
            if patch.start_line <= len(lines):
                del lines[patch.start_line - 1]

        elif patch.operation == PatchOperation.DELETE_BLOCK:
            # Delete block of lines
            start_idx = patch.start_line - 1
            end_idx = patch.end_line if patch.end_line else patch.start_line
            del lines[start_idx:end_idx]

        return "".join(lines)

    async def _validate_patched_file(
        self, file_path: str, content: str, dry_run: bool = False
    ) -> PatchValidationResult:
        """Validate the patched file for syntax and semantic correctness."""
        validation_errors = []
        warnings = []
        syntax_valid = True
        semantic_changes = []

        try:
            # Basic syntax validation for TypeScript/JavaScript
            if file_path.endswith(".ts"):
                syntax_valid = await self._validate_typescript_syntax(content, dry_run)
            elif file_path.endswith(".js"):
                syntax_valid = await self._validate_javascript_syntax(content, dry_run)

            if not syntax_valid:
                validation_errors.append("Syntax validation failed")

            # Check for common issues
            semantic_issues = self._check_semantic_issues(content)
            semantic_changes.extend(semantic_issues)

            # Check for Playwright best practices
            playwright_warnings = self._check_playwright_best_practices(content)
            warnings.extend(playwright_warnings)

        except Exception as e:
            validation_errors.append(f"Validation error: {e}")
            syntax_valid = False

        return PatchValidationResult(
            is_valid=len(validation_errors) == 0,
            validation_errors=validation_errors,
            warnings=warnings,
            syntax_valid=syntax_valid,
            semantic_changes=semantic_changes,
        )

    async def _validate_typescript_syntax(self, content: str, dry_run: bool) -> bool:
        """Validate TypeScript syntax using tsc."""
        if dry_run:
            # Skip actual compilation in dry run
            return True

        try:
            # Create temporary file for validation
            temp_file = Path("/tmp/temp_validation.ts")
            temp_file.write_text(content)

            # Run TypeScript compiler for syntax check
            result = subprocess.run(
                ["npx", "tsc", "--noEmit", "--skipLibCheck", str(temp_file)],
                capture_output=True,
                text=True,
                timeout=self.validation_timeout,
            )

            # Clean up
            temp_file.unlink(missing_ok=True)

            return result.returncode == 0

        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("TypeScript validation skipped - tsc not available")
            return True  # Assume valid if we can't validate
        except Exception as e:
            logger.warning(f"TypeScript validation failed: {e}")
            return False

    async def _validate_javascript_syntax(self, content: str, dry_run: bool) -> bool:
        """Validate JavaScript syntax using node."""
        if dry_run:
            return True

        try:
            # Use node to check syntax
            result = subprocess.run(
                ["node", "--check", "-"],
                input=content,
                capture_output=True,
                text=True,
                timeout=self.validation_timeout,
            )

            return result.returncode == 0

        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("JavaScript validation skipped - node not available")
            return True
        except Exception as e:
            logger.warning(f"JavaScript validation failed: {e}")
            return False

    def _check_semantic_issues(self, content: str) -> List[str]:
        """Check for semantic issues in the code."""
        issues = []

        # Check for unreachable code
        if re.search(r"return\s*;.*\n.*\S", content):
            issues.append("Potential unreachable code after return statement")

        # Check for unused variables (basic check)
        variable_declarations = re.findall(r"(?:const|let|var)\s+(\w+)", content)
        for var in variable_declarations:
            if content.count(var) == 1:  # Only declared, never used
                issues.append(f"Potentially unused variable: {var}")

        return issues

    def _check_playwright_best_practices(self, content: str) -> List[str]:
        """Check for Playwright best practices violations."""
        warnings = []

        # Check for hard-coded waits
        if re.search(r"page\.waitForTimeout\(\d+\)", content):
            warnings.append(
                "Hard-coded timeout found - consider using element-based waits"
            )

        # Check for CSS selectors instead of semantic selectors
        css_selectors = re.findall(r'page\.locator\([\'"]([^\'\"]*)[\'\"]\)', content)
        for selector in css_selectors:
            if selector.startswith(".") or selector.startswith("#") or " " in selector:
                warnings.append(
                    f"CSS selector found: {selector} - consider semantic selectors"
                )

        # Check for missing error handling
        if "await page." in content and "try" not in content:
            warnings.append("Consider adding error handling for page interactions")

        return warnings

    def _calculate_lines_changed(self, original: str, patched: str) -> int:
        """Calculate the number of lines changed between original and patched content."""
        original_lines = set(original.splitlines())
        patched_lines = set(patched.splitlines())

        # Lines that are different
        changed_lines = original_lines.symmetric_difference(patched_lines)
        return len(changed_lines)

    async def _rollback_patch(self, patch_result: PatchResult) -> None:
        """Internal method to rollback a patch."""
        if patch_result.backup_path:
            await self.rollback_patch(patch_result)

    async def _determine_patch_operation(
        self, fix_suggestion: FixSuggestion, content: str
    ) -> Tuple[PatchOperation, int, Optional[int]]:
        """Determine the patch operation based on fix suggestion."""

        if fix_suggestion.line_number:
            start_line = fix_suggestion.line_number
        else:
            # Try to find the line based on original code
            start_line = self._find_code_line(content, fix_suggestion.original_code)

        # Determine operation type
        if fix_suggestion.fix_type in [
            FixType.SELECTOR_REPLACEMENT,
            FixType.WAIT_CONDITION,
            FixType.ASSERTION_UPDATE,
        ]:
            if fix_suggestion.original_code and "\n" in fix_suggestion.original_code:
                # Multi-line replacement
                end_line = start_line + fix_suggestion.original_code.count("\n")
                return PatchOperation.REPLACE_BLOCK, start_line, end_line
            else:
                # Single line replacement
                return PatchOperation.REPLACE_LINE, start_line, None

        elif fix_suggestion.fix_type == FixType.ERROR_HANDLING:
            # Usually involves wrapping existing code
            return PatchOperation.REPLACE_BLOCK, start_line, start_line + 2

        else:
            # Default to single line replacement
            return PatchOperation.REPLACE_LINE, start_line, None

    def _find_code_line(self, content: str, target_code: Optional[str]) -> int:
        """Find the line number where target code appears."""
        if not target_code:
            return 1

        lines = content.splitlines()
        target_clean = target_code.strip()

        for i, line in enumerate(lines, 1):
            if target_clean in line.strip():
                return i

        return 1  # Default to first line if not found

    def _extract_original_code(
        self,
        content: str,
        start_line: int,
        end_line: Optional[int],
        operation: PatchOperation,
    ) -> str:
        """Extract the original code that will be replaced."""
        lines = content.splitlines()

        if operation == PatchOperation.REPLACE_LINE:
            if start_line <= len(lines):
                return lines[start_line - 1]

        elif operation == PatchOperation.REPLACE_BLOCK and end_line:
            start_idx = start_line - 1
            end_idx = min(end_line, len(lines))
            return "\n".join(lines[start_idx:end_idx])

        return ""

    async def _generate_patched_code(
        self,
        fix_suggestion: FixSuggestion,
        original_code: str,
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Generate the patched code based on fix suggestion."""

        # If suggestion already has suggested code, use it
        if fix_suggestion.suggested_code:
            return fix_suggestion.suggested_code

        # Generate code based on fix type
        if fix_suggestion.fix_type == FixType.WAIT_CONDITION:
            return self._generate_wait_condition_fix(original_code)

        elif fix_suggestion.fix_type == FixType.SELECTOR_REPLACEMENT:
            return self._generate_selector_replacement_fix(original_code)

        elif fix_suggestion.fix_type == FixType.TIMEOUT_ADJUSTMENT:
            return self._generate_timeout_adjustment_fix(original_code)

        elif fix_suggestion.fix_type == FixType.ERROR_HANDLING:
            return self._generate_error_handling_fix(original_code)

        else:
            # Fallback - return original with comment
            return (
                f"{original_code}  // TODO: Apply {fix_suggestion.fix_type.value} fix"
            )

    def _generate_wait_condition_fix(self, original_code: str) -> str:
        """Generate wait condition fix."""
        # Add explicit wait before action
        if ".click()" in original_code:
            element_part = original_code.split(".click()")[0]
            return f"await expect({element_part}).toBeVisible();\n  {original_code}"

        elif ".fill(" in original_code:
            element_part = original_code.split(".fill(")[0]
            return f"await expect({element_part}).toBeVisible();\n  {original_code}"

        else:
            return f"// Added wait condition\n  {original_code}"

    def _generate_selector_replacement_fix(self, original_code: str) -> str:
        """Generate selector replacement fix."""
        # Replace CSS selectors with semantic ones
        if "locator(" in original_code and (
            "#" in original_code or "." in original_code
        ):
            # Suggest getByTestId instead
            return (
                original_code.replace("locator(", "getByTestId(")
                .replace("#", "")
                .replace(".", "")
            )

        return original_code + "  // TODO: Replace with semantic selector"

    def _generate_timeout_adjustment_fix(self, original_code: str) -> str:
        """Generate timeout adjustment fix."""
        # Add timeout option to actions
        if ".click()" in original_code:
            return original_code.replace(".click()", ".click({ timeout: 10000 })")

        elif ".fill(" in original_code and ")" in original_code:
            # Add timeout to fill action
            return original_code.replace(")", ", { timeout: 10000 })")

        return original_code

    def _generate_error_handling_fix(self, original_code: str) -> str:
        """Generate error handling fix."""
        return f"""try {{
    {original_code}
  }} catch (error) {{
    console.log('Action failed, retrying...', error.message);
    await page.waitForTimeout(1000);
    {original_code}
  }}"""

    async def _analyze_flaky_patterns(self, content: str, test_name: str) -> List[str]:
        """Analyze test content for flaky patterns."""
        patterns = []

        # Check for hard-coded waits
        if re.search(r"waitForTimeout\(\d+\)", content):
            patterns.append("hard_coded_waits")

        # Check for race conditions
        if re.search(r"Promise\.all\(", content):
            patterns.append("potential_race_conditions")

        # Check for network dependencies
        if re.search(r"fetch\(|axios\.|request\(", content):
            patterns.append("network_dependencies")

        # Check for time-based logic
        if re.search(r"Date\.|setTimeout|setInterval", content):
            patterns.append("time_based_logic")

        # Check for external file dependencies
        if re.search(r"readFile|writeFile|fs\.", content):
            patterns.append("file_system_dependencies")

        return patterns

    def _calculate_flakiness_score(
        self, patterns: List[str], failure_history: Optional[List[Dict[str, Any]]]
    ) -> float:
        """Calculate flakiness score based on patterns and history."""
        score = 0.0

        # Pattern-based scoring
        pattern_weights = {
            "hard_coded_waits": 0.3,
            "potential_race_conditions": 0.4,
            "network_dependencies": 0.2,
            "time_based_logic": 0.3,
            "file_system_dependencies": 0.2,
        }

        for pattern in patterns:
            score += pattern_weights.get(pattern, 0.1)

        # History-based scoring
        if failure_history:
            failure_rate = sum(
                1 for run in failure_history if not run.get("success", True)
            )
            failure_rate /= len(failure_history)
            score += failure_rate * 0.5

        return min(score, 1.0)  # Cap at 1.0

    async def _generate_stability_suggestions(
        self, content: str, patterns: List[str], test_name: str
    ) -> List[FixSuggestion]:
        """Generate suggestions to improve test stability."""
        suggestions = []

        if "hard_coded_waits" in patterns:
            suggestions.append(
                FixSuggestion(
                    fix_type=FixType.WAIT_CONDITION,
                    description="Replace hard-coded waits with element-based waits",
                    confidence=ConfidenceLevel.HIGH,
                    reasoning="Element-based waits are more reliable than fixed timeouts",
                    potential_side_effects=["May need to adjust wait conditions"],
                    test_impact="minimal",
                )
            )

        if "potential_race_conditions" in patterns:
            suggestions.append(
                FixSuggestion(
                    fix_type=FixType.RETRY_LOGIC,
                    description="Add retry logic for race condition prone operations",
                    confidence=ConfidenceLevel.MEDIUM,
                    reasoning="Retry logic can handle timing-sensitive operations",
                    potential_side_effects=["Increased test execution time"],
                    test_impact="moderate",
                )
            )

        if "network_dependencies" in patterns:
            suggestions.append(
                FixSuggestion(
                    fix_type=FixType.ERROR_HANDLING,
                    description="Add error handling for network operations",
                    confidence=ConfidenceLevel.HIGH,
                    reasoning="Network operations can fail intermittently",
                    potential_side_effects=["More complex error handling logic"],
                    test_impact="minimal",
                )
            )

        return suggestions
