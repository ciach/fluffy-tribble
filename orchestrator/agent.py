"""
Main Agent Controller for QA Operator.

Orchestrates the complete testing workflow using the Agents SDK pattern,
coordinating planning, generation, execution, analysis, and patching components.
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from .core.config import Config
from .core.logging_config import get_logger, log_performance
from .core.workflow import WorkflowManager, WorkflowContext
from .core.exceptions import QAOperatorError, ValidationError, MCPConnectionError

from .mcp.connection_manager import MCPConnectionManager
from .mcp.playwright_client import PlaywrightMCPClient
from .mcp.filesystem_client import FilesystemMCPClient

from .models.router import ModelRouter
from .planning.engine import PlanningEngine
from .generation.generator import TestGenerator
from .generation.selector_auditor import SelectorAuditor
from .execution.executor import TestExecutor
from .execution.artifacts import ArtifactManager
from .analysis.analyzer import FailureAnalyzer
from .analysis.patcher import CodePatcher

from .planning.models import TestSpecification
from .execution.models import ExecutionResult, TestStatus
from .analysis.models import FailureAnalysis

from enum import Enum
from dataclasses import dataclass, field


class WorkflowState(Enum):
    """Workflow execution states."""
    INITIALIZING = "initializing"
    PLANNING = "planning"
    GENERATING = "generating"
    EXECUTING = "executing"
    ANALYZING = "analyzing"
    PATCHING = "patching"
    COMPLETED = "completed"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class WorkflowStateManager:
    """Manages workflow state and recovery information."""
    current_state: WorkflowState = WorkflowState.INITIALIZING
    completed_phases: List[str] = field(default_factory=list)
    failed_phases: List[str] = field(default_factory=list)
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    state_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def transition_to(self, new_state: WorkflowState, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Transition to a new workflow state."""
        old_state = self.current_state
        self.current_state = new_state
        
        # Record state transition
        transition_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "from_state": old_state.value,
            "to_state": new_state.value,
            "metadata": metadata or {}
        }
        self.state_history.append(transition_record)
    
    def mark_phase_completed(self, phase: str) -> None:
        """Mark a phase as completed."""
        if phase not in self.completed_phases:
            self.completed_phases.append(phase)
    
    def mark_phase_failed(self, phase: str) -> None:
        """Mark a phase as failed."""
        if phase not in self.failed_phases:
            self.failed_phases.append(phase)
    
    def can_recover(self) -> bool:
        """Check if recovery is possible."""
        return self.recovery_attempts < self.max_recovery_attempts
    
    def attempt_recovery(self) -> None:
        """Record a recovery attempt."""
        self.recovery_attempts += 1
        self.transition_to(WorkflowState.RECOVERING)
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current workflow state."""
        return {
            "current_state": self.current_state.value,
            "completed_phases": self.completed_phases,
            "failed_phases": self.failed_phases,
            "recovery_attempts": self.recovery_attempts,
            "can_recover": self.can_recover(),
            "state_transitions": len(self.state_history)
        }


class QAOperatorAgent:
    """
    Main Agent Controller for QA Operator.
    
    Orchestrates the complete testing workflow including planning, generation,
    execution, analysis, and patching using a modular component architecture.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the QA Operator Agent.
        
        Args:
            config: Optional configuration object
        """
        self.config = config or Config.from_env()
        self.logger = get_logger("qa_operator.agent")
        
        # Initialize workflow manager
        self.workflow_manager = WorkflowManager(self.config)
        self._current_workflow: Optional[WorkflowContext] = None
        
        # Component instances (initialized in setup)
        self.mcp_manager: Optional[MCPConnectionManager] = None
        self.playwright_client: Optional[PlaywrightMCPClient] = None
        self.filesystem_client: Optional[FilesystemMCPClient] = None
        self.model_router: Optional[ModelRouter] = None
        self.planning_engine: Optional[PlanningEngine] = None
        self.test_generator: Optional[TestGenerator] = None
        self.test_executor: Optional[TestExecutor] = None
        self.artifact_manager: Optional[ArtifactManager] = None
        self.failure_analyzer: Optional[FailureAnalyzer] = None
        self.code_patcher: Optional[CodePatcher] = None
        
        # Performance metrics
        self._metrics: Dict[str, Any] = {}
        self._component_timings: Dict[str, float] = {}
        
        # Workflow state management
        self._state_manager: Optional[WorkflowStateManager] = None

    async def initialize(self) -> None:
        """
        Initialize all agent components and establish connections.
        
        Raises:
            QAOperatorError: If initialization fails
        """
        start_time = time.time()
        
        try:
            self.logger.info("Initializing QA Operator Agent")
            
            # Initialize MCP connections
            await self._initialize_mcp_connections()
            
            # Initialize AI model routing
            self._initialize_model_router()
            
            # Initialize core components
            self._initialize_components()
            
            # Validate component readiness
            await self._validate_component_readiness()
            
            initialization_time = time.time() - start_time
            self._metrics["initialization_time"] = initialization_time
            
            self.logger.info(
                f"Agent initialization completed in {initialization_time:.2f}s",
                extra={
                    "metadata": {
                        "initialization_time": initialization_time,
                        "components_initialized": len([c for c in [
                            self.mcp_manager, self.model_router, self.planning_engine,
                            self.test_generator, self.test_executor, self.failure_analyzer,
                            self.code_patcher
                        ] if c is not None])
                    }
                }
            )
            
        except Exception as e:
            self.logger.error(
                f"Agent initialization failed: {str(e)}",
                extra={
                    "metadata": {
                        "error_type": e.__class__.__name__,
                        "initialization_time": time.time() - start_time
                    }
                }
            )
            raise QAOperatorError(f"Agent initialization failed: {str(e)}") from e

    async def _initialize_mcp_connections(self) -> None:
        """Initialize MCP server connections."""
        self.logger.info("Initializing MCP connections")
        
        # Initialize MCP connection manager
        self.mcp_manager = MCPConnectionManager(self.config.mcp_config_path)
        
        # Connect to required MCP servers
        await self.mcp_manager.connect_all()
        
        # Initialize MCP clients
        self.playwright_client = PlaywrightMCPClient(
            self.mcp_manager, 
            logger=self.logger
        )
        
        self.filesystem_client = FilesystemMCPClient(
            self.mcp_manager,
            e2e_dir=Path("e2e"),
            logger=self.logger
        )
        
        self.logger.info("MCP connections established")

    def _initialize_model_router(self) -> None:
        """Initialize AI model routing."""
        self.logger.info("Initializing model router")
        
        self.model_router = ModelRouter(
            config=self.config,
            logger=self.logger
        )
        
        self.logger.info("Model router initialized")

    def _initialize_components(self) -> None:
        """Initialize core workflow components."""
        self.logger.info("Initializing workflow components")
        
        # Initialize planning engine
        self.planning_engine = PlanningEngine(self.model_router)
        
        # Initialize test generator with selector auditor
        selector_auditor = SelectorAuditor()
        self.test_generator = TestGenerator(
            self.model_router,
            selector_auditor=selector_auditor
        )
        
        # Initialize artifact manager
        self.artifact_manager = ArtifactManager(
            config=self.config,
            workflow_id=None  # Will be set when workflow starts
        )
        
        # Initialize test executor
        self.test_executor = TestExecutor(
            config=self.config,
            playwright_client=self.playwright_client,
            artifact_manager=self.artifact_manager
        )
        
        # Initialize failure analyzer
        self.failure_analyzer = FailureAnalyzer(
            config=self.config,
            model_router=self.model_router
        )
        
        # Initialize code patcher
        self.code_patcher = CodePatcher(
            config=self.config,
            filesystem_client=self.filesystem_client
        )
        
        self.logger.info("Workflow components initialized")

    async def _validate_component_readiness(self) -> None:
        """Validate that all components are ready for operation."""
        self.logger.info("Validating component readiness")
        
        # Check MCP connections
        if not await self.mcp_manager.health_check():
            raise QAOperatorError("MCP servers not ready")
        
        # Check model router
        if not self.model_router.is_ready():
            raise QAOperatorError("Model router not ready")
        
        self.logger.info("All components ready")

    async def run_workflow(
        self, 
        specification: Union[str, Dict[str, Any], TestSpecification],
        workflow_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete testing workflow.
        
        Args:
            specification: Test specification (string, dict, or TestSpecification object)
            workflow_metadata: Optional metadata for the workflow
            
        Returns:
            Workflow execution results
            
        Raises:
            QAOperatorError: If workflow execution fails
        """
        # Start workflow
        self._current_workflow = self.workflow_manager.start_workflow(
            metadata=workflow_metadata or {}
        )
        
        workflow_id = self._current_workflow.workflow_id
        self.logger.info(f"Starting workflow: {workflow_id}")
        
        # Initialize state manager
        self._state_manager = WorkflowStateManager()
        self._state_manager.transition_to(WorkflowState.INITIALIZING)
        
        # Update artifact manager with workflow ID
        if self.artifact_manager:
            self.artifact_manager.workflow_id = workflow_id
        
        try:
            # Execute workflow phases with error handling and recovery
            results = await self._execute_workflow_phases_with_recovery(specification)
            
            # Log performance metrics
            self._log_performance_metrics()
            
            # End workflow successfully
            self.workflow_manager.end_workflow(success=True)
            
            return results
            
        except Exception as e:
            self.logger.error(
                f"Workflow failed: {str(e)}",
                extra={
                    "metadata": {
                        "workflow_id": workflow_id,
                        "error_type": e.__class__.__name__,
                        "duration": self._current_workflow.duration if self._current_workflow else 0
                    }
                }
            )
            
            # Attempt recovery if possible
            await self._attempt_workflow_recovery(e)
            
            # End workflow with failure
            self.workflow_manager.end_workflow(success=False, error=e)
            
            raise QAOperatorError(f"Workflow execution failed: {str(e)}") from e

    async def _execute_workflow_phases_with_recovery(
        self, 
        specification: Union[str, Dict[str, Any], TestSpecification]
    ) -> Dict[str, Any]:
        """Execute workflow phases with comprehensive error handling and recovery."""
        results = {
            "workflow_id": self._current_workflow.workflow_id,
            "phases": {},
            "metrics": {},
            "recovery_attempts": []
        }
        
        # Phase 1: Planning (with retry logic)
        try:
            self.logger.info("Phase 1: Test Planning")
            self._state_manager.transition_to(WorkflowState.PLANNING)
            phase_start = time.time()
            
            test_plan = await self._execute_planning_phase_with_retry(specification)
            self._state_manager.mark_phase_completed("planning")
            
            results["phases"]["planning"] = {
                "test_plan": test_plan.to_dict() if hasattr(test_plan, 'to_dict') else str(test_plan),
                "duration": time.time() - phase_start,
                "status": "success"
            }
            
        except Exception as e:
            self._state_manager.mark_phase_failed("planning")
            results["phases"]["planning"] = {
                "status": "failed",
                "error": str(e),
                "duration": time.time() - phase_start
            }
            # Planning failure is critical - cannot continue
            raise QAOperatorError(f"Planning phase failed: {str(e)}") from e
        
        # Phase 2: Test Generation (with retry logic)
        try:
            self.logger.info("Phase 2: Test Generation")
            self._state_manager.transition_to(WorkflowState.GENERATING)
            phase_start = time.time()
            
            generated_tests = await self._execute_generation_phase_with_retry(test_plan)
            self._state_manager.mark_phase_completed("generation")
            
            results["phases"]["generation"] = {
                "tests_generated": len(generated_tests),
                "test_files": [test.file_path for test in generated_tests],
                "duration": time.time() - phase_start,
                "status": "success"
            }
            
        except Exception as e:
            self._state_manager.mark_phase_failed("generation")
            results["phases"]["generation"] = {
                "status": "failed",
                "error": str(e),
                "duration": time.time() - phase_start
            }
            # Generation failure is critical - cannot continue
            raise QAOperatorError(f"Generation phase failed: {str(e)}") from e
        
        # Phase 3: Test Execution (with partial failure handling)
        try:
            self.logger.info("Phase 3: Test Execution")
            self._state_manager.transition_to(WorkflowState.EXECUTING)
            phase_start = time.time()
            
            execution_results = await self._execute_execution_phase_with_recovery(generated_tests)
            from .execution.models import TestStatus
            
            self._state_manager.mark_phase_completed("execution")
            
            results["phases"]["execution"] = {
                "tests_run": len(execution_results),
                "passed": len([r for r in execution_results if r.status == TestStatus.PASSED]),
                "failed": len([r for r in execution_results if r.status == TestStatus.FAILED]),
                "duration": time.time() - phase_start,
                "status": "success"
            }
            
        except Exception as e:
            self._state_manager.mark_phase_failed("execution")
            results["phases"]["execution"] = {
                "status": "failed",
                "error": str(e),
                "duration": time.time() - phase_start
            }
            # Execution failure is not critical - we can still analyze what we have
            self.logger.warning(f"Execution phase failed, continuing with available results: {str(e)}")
            execution_results = []
        
        # Phase 4: Failure Analysis and Patching (if needed)
        failed_tests = [r for r in execution_results if hasattr(r, 'status') and r.status == TestStatus.FAILED]
        if failed_tests:
            try:
                self.logger.info("Phase 4: Failure Analysis and Patching")
                self._state_manager.transition_to(WorkflowState.ANALYZING)
                phase_start = time.time()
                
                patch_results = await self._execute_analysis_and_patching_phase_with_recovery(failed_tests)
                self._state_manager.mark_phase_completed("analysis_patching")
                
                results["phases"]["analysis_patching"] = {
                    "failures_analyzed": len(failed_tests),
                    "patches_applied": len(patch_results),
                    "duration": time.time() - phase_start,
                    "status": "success"
                }
                
            except Exception as e:
                self._state_manager.mark_phase_failed("analysis_patching")
                results["phases"]["analysis_patching"] = {
                    "status": "failed",
                    "error": str(e),
                    "duration": time.time() - phase_start
                }
                # Analysis/patching failure is not critical - workflow can still complete
                self.logger.warning(f"Analysis and patching phase failed: {str(e)}")
        
        # Mark workflow as completed
        self._state_manager.transition_to(WorkflowState.COMPLETED)
        
        # Add state information to results
        results["workflow_state"] = self._state_manager.get_state_summary()
        
        return results

    async def _execute_planning_phase_with_retry(
        self, 
        specification: Union[str, Dict[str, Any], TestSpecification],
        max_retries: int = 2
    ) -> Any:
        """Execute planning phase with retry logic."""
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                return await self._execute_planning_phase(specification)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    self.logger.warning(
                        f"Planning attempt {attempt + 1} failed, retrying: {str(e)}"
                    )
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"Planning failed after {max_retries + 1} attempts")
        
        raise last_error

    async def _execute_generation_phase_with_retry(
        self, 
        test_plan: Any,
        max_retries: int = 2
    ) -> List[Any]:
        """Execute generation phase with retry logic."""
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                return await self._execute_generation_phase(test_plan)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    self.logger.warning(
                        f"Generation attempt {attempt + 1} failed, retrying: {str(e)}"
                    )
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"Generation failed after {max_retries + 1} attempts")
        
        raise last_error

    async def _execute_execution_phase_with_recovery(
        self, 
        generated_tests: List[Any]
    ) -> List[ExecutionResult]:
        """Execute execution phase with partial failure recovery."""
        execution_results = []
        
        for test in generated_tests:
            try:
                test_file = test.file_path if hasattr(test, 'file_path') else str(test)
                result = await self.test_executor.execute_test(test_file)
                execution_results.append(result)
            except Exception as e:
                self.logger.warning(f"Test execution failed for {test}: {str(e)}")
                # Create a failed result for tracking
                from .execution.models import ExecutionResult, TestStatus
                failed_result = ExecutionResult(
                    test_name=str(test),
                    test_file=test.file_path if hasattr(test, 'file_path') else str(test),
                    status=TestStatus.FAILED,
                    duration=0.0,
                    error_message=str(e),
                    artifacts={}
                )
                execution_results.append(failed_result)
        
        return execution_results

    async def _execute_analysis_and_patching_phase_with_recovery(
        self, 
        failed_tests: List[ExecutionResult]
    ) -> List[Any]:
        """Execute analysis and patching phase with recovery."""
        patch_results = []
        
        for failed_test in failed_tests:
            try:
                # Analyze failure
                analysis = await self.failure_analyzer.analyze_failure(failed_test)
                
                # Apply patches if analysis suggests fixes
                if hasattr(analysis, 'suggested_fixes') and analysis.suggested_fixes:
                    for fix in analysis.suggested_fixes:
                        try:
                            # Create patch from fix suggestion
                            patch = await self.code_patcher.create_patch_from_suggestion(
                                fix,
                                failed_test.test_file
                            )
                            
                            # Apply the patch
                            patch_result = await self.code_patcher.apply_patch(patch)
                            patch_results.append(patch_result)
                            
                            # Re-execute test after patch
                            if patch_result.success:
                                try:
                                    rerun_result = await self.test_executor.execute_test(
                                        failed_test.test_file
                                    )
                                    
                                    from .execution.models import TestStatus
                                    if rerun_result.status == TestStatus.PASSED:
                                        self.logger.info(
                                            f"Test fixed successfully: {failed_test.test_name}"
                                        )
                                    else:
                                        self.logger.warning(
                                            f"Test still failing after patch: {failed_test.test_name}"
                                        )
                                except Exception as re_exec_error:
                                    self.logger.warning(
                                        f"Re-execution failed for {failed_test.test_name}: {str(re_exec_error)}"
                                    )
                        except Exception as patch_error:
                            self.logger.warning(
                                f"Patch application failed for {failed_test.test_name}: {str(patch_error)}"
                            )
                            
            except Exception as analysis_error:
                self.logger.warning(
                    f"Analysis failed for {failed_test.test_name}: {str(analysis_error)}"
                )
        
        return patch_results

    async def _attempt_workflow_recovery(self, error: Exception) -> None:
        """Attempt to recover from workflow failure."""
        if not self._state_manager or not self._state_manager.can_recover():
            self.logger.warning("Recovery not possible or state manager not available")
            return
        
        self._state_manager.attempt_recovery()
        self.logger.info(f"Attempting workflow recovery (attempt {self._state_manager.recovery_attempts})")
        
        try:
            # Check MCP connections and attempt reconnection
            if self.mcp_manager:
                health_status = await self.mcp_manager.health_check()
                if not health_status:
                    self.logger.info("Attempting MCP reconnection")
                    await self.mcp_manager.connect_all()
            
            # Check model router status
            if self.model_router and not self.model_router.is_ready():
                self.logger.info("Reinitializing model router")
                self._initialize_model_router()
            
            # Validate component readiness after recovery
            await self._validate_component_readiness()
            
            self.logger.info("Recovery attempt completed successfully")
            
        except Exception as recovery_error:
            self.logger.error(f"Recovery attempt failed: {str(recovery_error)}")
            if self._state_manager:
                self._state_manager.transition_to(WorkflowState.FAILED)

    async def _execute_workflow_phases(
        self, 
        specification: Union[str, Dict[str, Any], TestSpecification]
    ) -> Dict[str, Any]:
        """Execute the main workflow phases."""
        results = {
            "workflow_id": self._current_workflow.workflow_id,
            "phases": {},
            "metrics": {}
        }
        
        # Phase 1: Planning
        self.logger.info("Phase 1: Test Planning")
        phase_start = time.time()
        
        test_plan = await self._execute_planning_phase(specification)
        results["phases"]["planning"] = {
            "test_plan": test_plan.to_dict() if hasattr(test_plan, 'to_dict') else str(test_plan),
            "duration": time.time() - phase_start
        }
        
        # Phase 2: Test Generation
        self.logger.info("Phase 2: Test Generation")
        phase_start = time.time()
        
        generated_tests = await self._execute_generation_phase(test_plan)
        results["phases"]["generation"] = {
            "tests_generated": len(generated_tests),
            "test_files": [test.file_path for test in generated_tests],
            "duration": time.time() - phase_start
        }
        
        # Phase 3: Test Execution
        self.logger.info("Phase 3: Test Execution")
        phase_start = time.time()
        
        execution_results = await self._execute_execution_phase(generated_tests)
        results["phases"]["execution"] = {
            "tests_run": len(execution_results),
            "passed": len([r for r in execution_results if r.status == TestStatus.PASSED]),
            "failed": len([r for r in execution_results if r.status == TestStatus.FAILED]),
            "duration": time.time() - phase_start
        }
        
        # Phase 4: Failure Analysis and Patching (if needed)
        failed_tests = [r for r in execution_results if r.status == TestStatus.FAILED]
        if failed_tests:
            self.logger.info("Phase 4: Failure Analysis and Patching")
            phase_start = time.time()
            
            patch_results = await self._execute_analysis_and_patching_phase(failed_tests)
            results["phases"]["analysis_patching"] = {
                "failures_analyzed": len(failed_tests),
                "patches_applied": len(patch_results),
                "duration": time.time() - phase_start
            }
        
        return results

    async def _execute_planning_phase(
        self, 
        specification: Union[str, Dict[str, Any], TestSpecification]
    ) -> Any:
        """Execute the planning phase."""
        start_time = time.time()
        
        try:
            # Convert specification to TestSpecification if needed
            if isinstance(specification, str):
                # Create a basic TestSpecification from string
                from .planning.models import TestSpecification, Priority
                import uuid
                test_spec = TestSpecification(
                    id=str(uuid.uuid4()),
                    name="Generated Test",
                    description=specification,
                    requirements=[specification],
                    priority=Priority.MEDIUM
                )
            elif isinstance(specification, dict):
                # Convert dict to TestSpecification
                from .planning.models import TestSpecification, Priority
                test_spec = TestSpecification(
                    id=specification.get("id", str(uuid.uuid4())),
                    name=specification.get("name", "Generated Test"),
                    description=specification.get("description", ""),
                    requirements=specification.get("requirements", []),
                    priority=Priority(specification.get("priority", "medium")),
                    tags=specification.get("tags", []),
                    metadata=specification.get("metadata", {})
                )
            else:
                test_spec = specification
            
            # Generate test plan
            test_plan = self.planning_engine.create_test_plan(test_spec)
            
            self._component_timings["planning"] = time.time() - start_time
            
            self.logger.info(
                "Planning phase completed",
                extra={
                    "metadata": {
                        "test_cases": len(test_plan.test_cases) if hasattr(test_plan, 'test_cases') else 0,
                        "duration": self._component_timings["planning"]
                    }
                }
            )
            
            return test_plan
            
        except Exception as e:
            self.logger.error(f"Planning phase failed: {str(e)}")
            raise

    async def _execute_generation_phase(self, test_plan: Any) -> List[Any]:
        """Execute the test generation phase."""
        start_time = time.time()
        
        try:
            generated_tests = []
            
            # Generate tests for each test case in the plan
            if hasattr(test_plan, 'test_cases') and test_plan.test_cases:
                for i, test_case in enumerate(test_plan.test_cases):
                    file_name = f"test_{test_case.name.lower().replace(' ', '_')}.spec.ts"
                    generated_test = self.test_generator.generate_test(
                        test_plan, 
                        file_name=file_name,
                        output_dir="e2e"
                    )
                    generated_tests.append(generated_test)
            else:
                # Fallback: generate single test from plan
                file_name = f"test_{test_plan.specification_id.replace('-', '_')}.spec.ts"
                generated_test = self.test_generator.generate_test(
                    test_plan,
                    file_name=file_name,
                    output_dir="e2e"
                )
                generated_tests.append(generated_test)
            
            self._component_timings["generation"] = time.time() - start_time
            
            self.logger.info(
                "Generation phase completed",
                extra={
                    "metadata": {
                        "tests_generated": len(generated_tests),
                        "duration": self._component_timings["generation"]
                    }
                }
            )
            
            return generated_tests
            
        except Exception as e:
            self.logger.error(f"Generation phase failed: {str(e)}")
            raise

    async def _execute_execution_phase(self, generated_tests: List[Any]) -> List[ExecutionResult]:
        """Execute the test execution phase."""
        start_time = time.time()
        
        try:
            execution_results = []
            
            for test in generated_tests:
                # Execute individual test using the file path
                test_file = test.file_path if hasattr(test, 'file_path') else str(test)
                result = await self.test_executor.execute_test(test_file)
                execution_results.append(result)
            
            self._component_timings["execution"] = time.time() - start_time
            
            # Import TestStatus from the correct module
            from .execution.models import TestStatus
            
            self.logger.info(
                "Execution phase completed",
                extra={
                    "metadata": {
                        "tests_executed": len(execution_results),
                        "passed": len([r for r in execution_results if r.status == TestStatus.PASSED]),
                        "failed": len([r for r in execution_results if r.status == TestStatus.FAILED]),
                        "duration": self._component_timings["execution"]
                    }
                }
            )
            
            return execution_results
            
        except Exception as e:
            self.logger.error(f"Execution phase failed: {str(e)}")
            raise

    async def _execute_analysis_and_patching_phase(
        self, 
        failed_tests: List[ExecutionResult]
    ) -> List[Any]:
        """Execute the failure analysis and patching phase."""
        start_time = time.time()
        
        try:
            patch_results = []
            
            for failed_test in failed_tests:
                # Analyze failure
                analysis = await self.failure_analyzer.analyze_failure(failed_test)
                
                # Apply patches if analysis suggests fixes
                if hasattr(analysis, 'suggested_fixes') and analysis.suggested_fixes:
                    for fix in analysis.suggested_fixes:
                        # Create patch from fix suggestion
                        patch = await self.code_patcher.create_patch_from_suggestion(
                            fix,
                            failed_test.test_file
                        )
                        
                        # Apply the patch
                        patch_result = await self.code_patcher.apply_patch(patch)
                        patch_results.append(patch_result)
                        
                        # Re-execute test after patch
                        if patch_result.success:
                            rerun_result = await self.test_executor.execute_test(
                                failed_test.test_file
                            )
                            
                            from .execution.models import TestStatus
                            if rerun_result.status == TestStatus.PASSED:
                                self.logger.info(
                                    f"Test fixed successfully: {failed_test.test_name}"
                                )
                            else:
                                self.logger.warning(
                                    f"Test still failing after patch: {failed_test.test_name}"
                                )
            
            self._component_timings["analysis_patching"] = time.time() - start_time
            
            self.logger.info(
                "Analysis and patching phase completed",
                extra={
                    "metadata": {
                        "failures_analyzed": len(failed_tests),
                        "patches_applied": len(patch_results),
                        "duration": self._component_timings["analysis_patching"]
                    }
                }
            )
            
            return patch_results
            
        except Exception as e:
            self.logger.error(f"Analysis and patching phase failed: {str(e)}")
            raise

    def _log_performance_metrics(self) -> None:
        """Log comprehensive performance metrics for the workflow."""
        if not self._current_workflow:
            return
        
        total_duration = self._current_workflow.duration
        
        # Calculate model usage metrics
        model_metrics = self.model_router.get_usage_metrics() if self.model_router else {}
        
        # Compile comprehensive metrics
        metrics = {
            "workflow_id": self._current_workflow.workflow_id,
            "total_duration": total_duration,
            "component_timings": self._component_timings,
            "model_usage": model_metrics,
            "initialization_time": self._metrics.get("initialization_time", 0),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.logger.info(
            f"Workflow performance metrics - Total: {total_duration:.2f}s",
            extra={"metadata": {"performance_metrics": metrics}}
        )
        
        # Store metrics for potential reporting
        self._metrics.update(metrics)

    async def shutdown(self) -> None:
        """
        Gracefully shutdown the agent and cleanup resources.
        """
        self.logger.info("Shutting down QA Operator Agent")
        
        try:
            # Cleanup MCP connections
            if self.mcp_manager:
                await self.mcp_manager.disconnect_all()
            
            # Cleanup artifact manager
            if self.artifact_manager:
                await self.artifact_manager.cleanup()
            
            self.logger.info("Agent shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")

    @property
    def is_ready(self) -> bool:
        """Check if the agent is ready for workflow execution."""
        return all([
            self.mcp_manager is not None,
            self.model_router is not None,
            self.planning_engine is not None,
            self.test_generator is not None,
            self.test_executor is not None,
            self.failure_analyzer is not None,
            self.code_patcher is not None
        ])

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and health information."""
        status = {
            "ready": self.is_ready,
            "current_workflow": self._current_workflow.workflow_id if self._current_workflow else None,
            "components": {
                "mcp_manager": self.mcp_manager is not None,
                "model_router": self.model_router is not None,
                "planning_engine": self.planning_engine is not None,
                "test_generator": self.test_generator is not None,
                "test_executor": self.test_executor is not None,
                "failure_analyzer": self.failure_analyzer is not None,
                "code_patcher": self.code_patcher is not None
            },
            "metrics": self._metrics
        }
        
        # Add workflow state information if available
        if self._state_manager:
            status["workflow_state"] = self._state_manager.get_state_summary()
        
        return status