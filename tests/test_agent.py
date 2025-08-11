"""
Unit tests for Agent Controller.

Tests main orchestration component that coordinates the complete testing workflow
using the Agents SDK pattern.
"""

import asyncio
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from orchestrator.agent import WorkflowState, WorkflowStateManager
from orchestrator.core.config import Config
from orchestrator.core.exceptions import (
    QAOperatorError,
    ValidationError,
    MCPConnectionError,
)
from orchestrator.planning.models import TestSpecification
from orchestrator.execution.models import ExecutionResult, TestStatus
from orchestrator.analysis.models import FailureAnalysis


class TestWorkflowState:
    """Test cases for WorkflowState enum."""

    def test_workflow_state_values(self):
        """Test workflow state enum values."""
        assert WorkflowState.INITIALIZING.value == "initializing"
        assert WorkflowState.PLANNING.value == "planning"
        assert WorkflowState.GENERATING.value == "generating"
        assert WorkflowState.EXECUTING.value == "executing"
        assert WorkflowState.ANALYZING.value == "analyzing"
        assert WorkflowState.PATCHING.value == "patching"
        assert WorkflowState.COMPLETED.value == "completed"


class TestWorkflowStateManager:
    """Test cases for WorkflowStateManager."""

    def test_initial_state(self):
        """Test initial workflow state manager."""
        manager = WorkflowStateManager()

        assert manager.current_state == WorkflowState.INITIALIZING
        assert manager.completed_phases == []
        assert manager.failed_phases == []
        assert manager.recovery_attempts == 0
        assert manager.max_recovery_attempts == 3

    def test_state_transition(self):
        """Test state transition."""
        manager = WorkflowStateManager()

        manager.transition_to(WorkflowState.PLANNING, {"test": "metadata"})

        assert manager.current_state == WorkflowState.PLANNING
        assert len(manager.state_history) == 1
        assert manager.state_history[0]["from_state"] == "initializing"
        assert manager.state_history[0]["to_state"] == "planning"
        assert manager.state_history[0]["metadata"]["test"] == "metadata"

    def test_multiple_transitions(self):
        """Test multiple state transitions."""
        manager = WorkflowStateManager()

        manager.transition_to(WorkflowState.PLANNING)
        manager.transition_to(WorkflowState.GENERATING)
        manager.transition_to(WorkflowState.EXECUTING)

        assert manager.current_state == WorkflowState.EXECUTING
        assert len(manager.state_history) == 3


class TestWorkflowStateIntegration:
    """Test cases for workflow state integration."""

    def test_workflow_state_enum_values(self):
        """Test workflow state enum values."""
        assert WorkflowState.INITIALIZING.value == "initializing"
        assert WorkflowState.PLANNING.value == "planning"
        assert WorkflowState.GENERATING.value == "generating"
        assert WorkflowState.EXECUTING.value == "executing"
        assert WorkflowState.ANALYZING.value == "analyzing"
        assert WorkflowState.PATCHING.value == "patching"
        assert WorkflowState.COMPLETED.value == "completed"
        assert WorkflowState.FAILED.value == "failed"
        assert WorkflowState.RECOVERING.value == "recovering"

    def test_state_manager_with_recovery(self):
        """Test state manager with recovery attempts."""
        manager = WorkflowStateManager()

        # Simulate failure and recovery
        manager.transition_to(WorkflowState.FAILED, {"error": "Connection timeout"})
        manager.recovery_attempts += 1
        manager.transition_to(WorkflowState.RECOVERING)
        manager.transition_to(WorkflowState.EXECUTING)

        assert manager.current_state == WorkflowState.EXECUTING
        assert manager.recovery_attempts == 1
        assert len(manager.state_history) == 3

    def test_max_recovery_attempts(self):
        """Test maximum recovery attempts handling."""
        manager = WorkflowStateManager()
        manager.max_recovery_attempts = 2

        # Simulate multiple recovery attempts
        for i in range(3):
            manager.recovery_attempts += 1
            manager.transition_to(WorkflowState.RECOVERING)

        assert manager.recovery_attempts == 3
        assert manager.recovery_attempts > manager.max_recovery_attempts

    def test_completed_phases_tracking(self):
        """Test tracking of completed phases."""
        manager = WorkflowStateManager()

        manager.completed_phases.append("planning")
        manager.completed_phases.append("generation")
        manager.transition_to(WorkflowState.EXECUTING)

        assert "planning" in manager.completed_phases
        assert "generation" in manager.completed_phases
        assert len(manager.completed_phases) == 2

    def test_failed_phases_tracking(self):
        """Test tracking of failed phases."""
        manager = WorkflowStateManager()

        manager.failed_phases.append("execution")
        manager.transition_to(WorkflowState.ANALYZING)

        assert "execution" in manager.failed_phases
        assert len(manager.failed_phases) == 1
