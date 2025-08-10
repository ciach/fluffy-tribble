"""
Workflow management for QA Operator.

Handles workflow ID generation, correlation, and basic workflow state management.
"""

import uuid
import time
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from .config import Config
from .logging_config import get_logger


@dataclass
class WorkflowContext:
    """Context information for a workflow execution."""

    workflow_id: str
    start_time: float = field(default_factory=time.time)
    config: Optional[Config] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Get current workflow duration in seconds."""
        return time.time() - self.start_time

    @property
    def start_timestamp(self) -> str:
        """Get formatted start timestamp."""
        return datetime.fromtimestamp(self.start_time).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow context to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "start_time": self.start_timestamp,
            "duration": self.duration,
            "metadata": self.metadata,
        }


class WorkflowManager:
    """Manages workflow execution and correlation."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.from_env()
        self.logger = get_logger("qa_operator.workflow")
        self._current_workflow: Optional[WorkflowContext] = None

    def generate_workflow_id(self) -> str:
        """
        Generate a unique workflow ID for correlating logs, artifacts, and Git commits.

        Returns:
            Unique workflow identifier
        """
        # Generate UUID4 and take first 8 characters for readability
        workflow_id = str(uuid.uuid4()).replace("-", "")[:16]

        # Add timestamp prefix for chronological sorting
        timestamp = datetime.utcnow().strftime("%Y%m%d")

        return f"{timestamp}-{workflow_id}"

    def start_workflow(
        self, metadata: Optional[Dict[str, Any]] = None
    ) -> WorkflowContext:
        """
        Start a new workflow execution.

        Args:
            metadata: Optional metadata to associate with the workflow

        Returns:
            Workflow context object
        """
        workflow_id = self.generate_workflow_id()

        context = WorkflowContext(
            workflow_id=workflow_id, config=self.config, metadata=metadata or {}
        )

        self._current_workflow = context

        self.logger.info(
            f"Workflow started: {workflow_id}",
            extra={
                "metadata": {
                    "workflow_id": workflow_id,
                    "start_time": context.start_timestamp,
                    "config": self.config.to_dict(),
                    **context.metadata,
                }
            },
        )

        return context

    def end_workflow(
        self, success: bool = True, error: Optional[Exception] = None
    ) -> None:
        """
        End the current workflow execution.

        Args:
            success: Whether the workflow completed successfully
            error: Optional error that caused workflow failure
        """
        if not self._current_workflow:
            self.logger.warning("Attempted to end workflow but no workflow is active")
            return

        context = self._current_workflow
        duration = context.duration

        log_data = {
            "metadata": {
                "workflow_id": context.workflow_id,
                "duration": duration,
                "success": success,
                **context.metadata,
            }
        }

        if error:
            log_data["metadata"]["error"] = str(error)
            log_data["metadata"]["error_type"] = error.__class__.__name__

        if success:
            self.logger.info(
                f"Workflow completed successfully: {context.workflow_id} ({duration:.2f}s)",
                extra=log_data,
            )
        else:
            self.logger.error(
                f"Workflow failed: {context.workflow_id} ({duration:.2f}s)",
                extra=log_data,
            )

        self._current_workflow = None

    @property
    def current_workflow(self) -> Optional[WorkflowContext]:
        """Get the current active workflow context."""
        return self._current_workflow

    @property
    def current_workflow_id(self) -> Optional[str]:
        """Get the current workflow ID."""
        return self._current_workflow.workflow_id if self._current_workflow else None

    def update_metadata(self, **metadata) -> None:
        """
        Update metadata for the current workflow.

        Args:
            **metadata: Key-value pairs to add to workflow metadata
        """
        if self._current_workflow:
            self._current_workflow.metadata.update(metadata)
        else:
            self.logger.warning(
                "Attempted to update workflow metadata but no workflow is active"
            )

    def get_artifact_path(self, artifact_name: str) -> str:
        """
        Get the artifact path for the current workflow.

        Args:
            artifact_name: Name of the artifact

        Returns:
            Full path for the artifact including workflow ID
        """
        if not self._current_workflow:
            raise RuntimeError("No active workflow for artifact path generation")

        workflow_id = self._current_workflow.workflow_id
        return str(self.config.artifacts_dir / workflow_id / artifact_name)

    def get_commit_message_prefix(self) -> str:
        """
        Get the commit message prefix with workflow ID for Git correlation.

        Returns:
            Commit message prefix including workflow ID
        """
        if not self._current_workflow:
            raise RuntimeError("No active workflow for commit message generation")

        return f"[QA-{self._current_workflow.workflow_id}]"
