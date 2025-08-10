"""
QA Operator main application entry point.

This module provides the main entry point for the QA Operator agent,
setting up the core infrastructure and workflow management.
"""

import sys
import traceback
from typing import Optional

from .core.config import Config
from .core.logging_config import setup_logging, get_logger
from .core.workflow import WorkflowManager
from .core.exceptions import QAOperatorError, ValidationError


def main() -> int:
    """
    Main entry point for QA Operator.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    workflow_manager = None

    try:
        # Load configuration
        config = Config.from_env()

        # Initialize workflow manager
        workflow_manager = WorkflowManager(config)

        # Start workflow
        workflow_context = workflow_manager.start_workflow(
            {"entry_point": "main", "args": sys.argv[1:] if len(sys.argv) > 1 else []}
        )

        # Set up logging with workflow ID
        setup_logging(config, workflow_context.workflow_id)
        logger = get_logger("qa_operator.main")

        # Validate configuration
        config.validate()

        logger.info(
            "QA Operator starting up",
            extra={
                "metadata": {
                    "workflow_id": workflow_context.workflow_id,
                    "config": config.to_dict(),
                }
            },
        )

        # TODO: Initialize and run the main agent workflow
        # This will be implemented in subsequent tasks
        logger.info("QA Operator infrastructure initialized successfully")

        # End workflow successfully
        workflow_manager.end_workflow(success=True)
        return 0

    except ValidationError as e:
        if workflow_manager:
            logger = get_logger("qa_operator.main")
            logger.error(
                f"Configuration validation failed: {e.message}",
                extra={"metadata": e.to_dict()},
            )
            workflow_manager.end_workflow(success=False, error=e)
        else:
            print(f"Configuration validation failed: {e.message}", file=sys.stderr)
        return 1

    except QAOperatorError as e:
        if workflow_manager:
            logger = get_logger("qa_operator.main")
            logger.error(
                f"QA Operator error: {e.message}", extra={"metadata": e.to_dict()}
            )
            workflow_manager.end_workflow(success=False, error=e)
        else:
            print(f"QA Operator error: {e.message}", file=sys.stderr)
        return 1

    except Exception as e:
        if workflow_manager:
            logger = get_logger("qa_operator.main")
            logger.error(
                f"Unexpected error: {str(e)}",
                extra={
                    "metadata": {
                        "error_type": e.__class__.__name__,
                        "traceback": traceback.format_exc(),
                    }
                },
            )
            workflow_manager.end_workflow(success=False, error=e)
        else:
            print(f"Unexpected error: {str(e)}", file=sys.stderr)
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
