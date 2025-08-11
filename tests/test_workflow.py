"""
Unit tests for workflow management.

Tests workflow ID generation, correlation, and basic workflow state management.
"""

import time
import uuid
from unittest.mock import patch, MagicMock
import pytest

from orchestrator.core.workflow import WorkflowContext, WorkflowManager
from orchestrator.core.config import Config
from orchestrator.core.exceptions import ValidationError


class TestWorkflowContext:
    """Test cases for WorkflowContext."""

    def test_workflow_context_creation(self):
        """Test creating workflow context with default values."""
        workflow_id = "test-workflow-123"
        context = WorkflowContext(workflow_id=workflow_id)

        assert context.workflow_id == workflow_id
        assert isinstance(context.start_time, float)
        assert context.config is None
        assert context.metadata == {}

    def test_workflow_context_with_config(self):
        """Test creating workflow context with config."""
        config = Config()
        workflow_id = "test-workflow-123"

        context = WorkflowContext(
            workflow_id=workflow_id, config=config, metadata={"test": "value"}
        )

        assert context.workflow_id == workflow_id
        assert context.config == config
        assert context.metadata["test"] == "value"

    def test_duration_property(self):
        """Test duration property calculation."""
        context = WorkflowContext(workflow_id="test-123")

        # Sleep briefly to ensure duration > 0
        time.sleep(0.01)

        duration = context.duration
        assert duration > 0
        assert isinstance(duration, float)

    def test_start_timestamp_property(self):
        """Test start timestamp formatting."""
        context = WorkflowContext(workflow_id="test-123")

        timestamp = context.start_timestamp
        assert isinstance(timestamp, str)
        assert "T" in timestamp  # ISO format should contain T
        assert len(timestamp) > 10  # Should be full timestamp

    def test_to_dict(self):
        """Test converting workflow context to dictionary."""
        context = WorkflowContext(workflow_id="test-123", metadata={"key": "value"})

        context_dict = context.to_dict()

        assert context_dict["workflow_id"] == "test-123"
        assert "start_time" in context_dict
        assert "duration" in context_dict
        assert context_dict["metadata"]["key"] == "value"


class TestWorkflowManager:
    """Test cases for WorkflowManager."""

    def test_workflow_manager_creation(self):
        """Test creating workflow manager."""
        config = Config()
        manager = WorkflowManager(config)

        assert manager.config == config
        assert manager.active_workflows == {}

    def test_workflow_manager_default_config(self):
        """Test creating workflow manager with default config."""
        with patch("orchestrator.core.workflow.Config.from_env") as mock_from_env:
            mock_config = Config()
            mock_from_env.return_value = mock_config

            manager = WorkflowManager()

            assert manager.config == mock_config
            mock_from_env.assert_called_once()

    def test_start_workflow(self):
        """Test starting a new workflow."""
        manager = WorkflowManager(Config())

        with patch("uuid.uuid4") as mock_uuid:
            mock_uuid.return_value.hex = "abc123def456"

            context = manager.start_workflow({"test": "data"})

            assert context.workflow_id.startswith("workflow-")
            assert context.metadata["test"] == "data"
            assert context.workflow_id in manager.active_workflows

    def test_start_workflow_with_custom_id(self):
        """Test starting workflow with custom ID."""
        manager = WorkflowManager(Config())

        context = manager.start_workflow(
            {"test": "data"}, workflow_id="custom-workflow-123"
        )

        assert context.workflow_id == "custom-workflow-123"
        assert context.metadata["test"] == "data"
        assert "custom-workflow-123" in manager.active_workflows

    def test_get_workflow(self):
        """Test getting existing workflow context."""
        manager = WorkflowManager(Config())

        # Start a workflow
        context = manager.start_workflow({"test": "data"})
        workflow_id = context.workflow_id

        # Retrieve the workflow
        retrieved_context = manager.get_workflow(workflow_id)

        assert retrieved_context == context
        assert retrieved_context.workflow_id == workflow_id

    def test_get_nonexistent_workflow(self):
        """Test getting non-existent workflow returns None."""
        manager = WorkflowManager(Config())

        context = manager.get_workflow("nonexistent-workflow")

        assert context is None

    def test_complete_workflow(self):
        """Test completing a workflow."""
        manager = WorkflowManager(Config())

        # Start a workflow
        context = manager.start_workflow({"test": "data"})
        workflow_id = context.workflow_id

        # Complete the workflow
        result = manager.complete_workflow(workflow_id, {"status": "success"})

        assert result["workflow_id"] == workflow_id
        assert result["status"] == "success"
        assert "duration" in result
        assert workflow_id not in manager.active_workflows

    def test_complete_nonexistent_workflow(self):
        """Test completing non-existent workflow raises error."""
        manager = WorkflowManager(Config())

        with pytest.raises(ValidationError, match="Workflow .* not found"):
            manager.complete_workflow("nonexistent-workflow", {})

    def test_list_active_workflows(self):
        """Test listing active workflows."""
        manager = WorkflowManager(Config())

        # Start multiple workflows
        context1 = manager.start_workflow({"test": "data1"})
        context2 = manager.start_workflow({"test": "data2"})

        active_workflows = manager.list_active_workflows()

        assert len(active_workflows) == 2
        workflow_ids = [w["workflow_id"] for w in active_workflows]
        assert context1.workflow_id in workflow_ids
        assert context2.workflow_id in workflow_ids

    def test_cleanup_stale_workflows(self):
        """Test cleaning up stale workflows."""
        manager = WorkflowManager(Config())

        # Start a workflow and manually set old start time
        context = manager.start_workflow({"test": "data"})
        context.start_time = time.time() - 7200  # 2 hours ago

        # Cleanup stale workflows (older than 1 hour)
        cleaned_count = manager.cleanup_stale_workflows(max_age_seconds=3600)

        assert cleaned_count == 1
        assert context.workflow_id not in manager.active_workflows

    def test_cleanup_no_stale_workflows(self):
        """Test cleanup when no stale workflows exist."""
        manager = WorkflowManager(Config())

        # Start a recent workflow
        context = manager.start_workflow({"test": "data"})

        # Cleanup stale workflows
        cleaned_count = manager.cleanup_stale_workflows(max_age_seconds=3600)

        assert cleaned_count == 0
        assert context.workflow_id in manager.active_workflows

    def test_get_workflow_metrics(self):
        """Test getting workflow metrics."""
        manager = WorkflowManager(Config())

        # Start workflows
        context1 = manager.start_workflow({"test": "data1"})
        context2 = manager.start_workflow({"test": "data2"})

        # Complete one workflow
        manager.complete_workflow(context1.workflow_id, {"status": "success"})

        metrics = manager.get_workflow_metrics()

        assert metrics["active_count"] == 1
        assert metrics["completed_count"] == 1
        assert metrics["total_count"] == 2

    def test_workflow_correlation_id_generation(self):
        """Test workflow correlation ID generation."""
        manager = WorkflowManager(Config())

        context = manager.start_workflow({})

        # Should generate a valid UUID-based workflow ID
        assert context.workflow_id.startswith("workflow-")
        assert len(context.workflow_id) > 20  # Should be reasonably long

    def test_workflow_metadata_update(self):
        """Test updating workflow metadata."""
        manager = WorkflowManager(Config())

        context = manager.start_workflow({"initial": "data"})

        # Update metadata
        manager.update_workflow_metadata(
            context.workflow_id, {"updated": "value", "status": "running"}
        )

        updated_context = manager.get_workflow(context.workflow_id)
        assert updated_context.metadata["initial"] == "data"
        assert updated_context.metadata["updated"] == "value"
        assert updated_context.metadata["status"] == "running"

    def test_update_nonexistent_workflow_metadata(self):
        """Test updating metadata for non-existent workflow."""
        manager = WorkflowManager(Config())

        with pytest.raises(ValidationError, match="Workflow .* not found"):
            manager.update_workflow_metadata("nonexistent", {"key": "value"})
