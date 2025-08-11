#!/usr/bin/env python3
"""
End-to-end validation runner for QA Operator.

This script runs comprehensive validation tests including:
- Sample specification validation
- Dry-run workflow testing
- Performance benchmarking
- Integration testing with mocked MCP servers
"""

import os
import sys
import json
import asyncio
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from orchestrator.core.config import Config
from orchestrator.planning.models import TestSpecification


class E2EValidationRunner:
    """Main runner for end-to-end validation tests."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {
            "start_time": datetime.now().isoformat(),
            "validation_results": {},
            "performance_metrics": {},
            "errors": [],
        }

    def run_validation(self) -> Dict[str, Any]:
        """Run complete end-to-end validation."""
        print("🚀 Starting QA Operator E2E Validation")
        print("=" * 50)

        try:
            # 1. Validate sample specifications
            if self.config.get("validate_specifications", True):
                self._validate_specifications()

            # 2. Run dry-run tests
            if self.config.get("run_dry_run", True):
                self._run_dry_run_tests()

            # 3. Run unit tests
            if self.config.get("run_unit_tests", True):
                self._run_unit_tests()

            # 4. Run integration tests
            if self.config.get("run_integration_tests", True):
                self._run_integration_tests()

            # 5. Run performance benchmarks
            if self.config.get("run_performance_tests", True):
                self._run_performance_tests()

            # 6. Generate final report
            self._generate_final_report()

        except Exception as e:
            self.results["errors"].append(f"Validation failed: {str(e)}")
            print(f"❌ Validation failed: {e}")

        finally:
            self.results["end_time"] = datetime.now().isoformat()

        return self.results

    def _validate_specifications(self):
        """Validate all sample test specifications."""
        print("\n📋 Validating Sample Specifications")
        print("-" * 30)

        specs_dir = project_root / "e2e" / "sample_specifications"
        if not specs_dir.exists():
            self.results["errors"].append("Sample specifications directory not found")
            print("❌ Sample specifications directory not found")
            return

        valid_specs = 0
        total_specs = 0
        spec_results = []

        for spec_file in specs_dir.glob("*.json"):
            total_specs += 1
            try:
                with open(spec_file) as f:
                    spec_data = json.load(f)

                # Validate specification structure
                spec = TestSpecification(**spec_data)

                # Additional validation checks
                validation_result = self._validate_specification_content(spec)

                if validation_result["valid"]:
                    print(f"✅ {spec.name}")
                    valid_specs += 1
                    spec_results.append(
                        {
                            "name": spec.name,
                            "file": spec_file.name,
                            "valid": True,
                            "requirements_count": len(spec.requirements),
                            "tags": spec.tags,
                        }
                    )
                else:
                    print(f"❌ {spec.name}: {validation_result['error']}")
                    spec_results.append(
                        {
                            "name": spec.name,
                            "file": spec_file.name,
                            "valid": False,
                            "error": validation_result["error"],
                        }
                    )

            except Exception as e:
                print(f"❌ {spec_file.name}: {str(e)}")
                spec_results.append(
                    {"file": spec_file.name, "valid": False, "error": str(e)}
                )

        self.results["validation_results"]["specifications"] = {
            "total": total_specs,
            "valid": valid_specs,
            "success_rate": valid_specs / total_specs if total_specs > 0 else 0,
            "details": spec_results,
        }

        print(
            f"\n📊 Specification Validation: {valid_specs}/{total_specs} valid ({valid_specs/total_specs*100:.1f}%)"
        )

    def _validate_specification_content(
        self, spec: TestSpecification
    ) -> Dict[str, Any]:
        """Validate specification content quality."""
        try:
            # Check required fields
            if not spec.name or not spec.description:
                return {"valid": False, "error": "Missing name or description"}

            if not spec.requirements or len(spec.requirements) == 0:
                return {"valid": False, "error": "No requirements specified"}

            # Check requirement quality
            for req in spec.requirements:
                if len(req.strip()) < 10:
                    return {"valid": False, "error": f"Requirement too short: '{req}'"}

                # More flexible requirement validation - just check for action words
                if not any(
                    word in req.lower()
                    for word in [
                        "user",
                        "system",
                        "can",
                        "should",
                        "must",
                        "will",
                        "shall",
                        "is",
                        "are",
                        "show",
                        "display",
                        "validate",
                        "prevent",
                        "allow",
                        "enforce",
                        "require",
                        "enable",
                        "provide",
                    ]
                ):
                    return {
                        "valid": False,
                        "error": f"Requirement lacks action words: '{req}'",
                    }

            # Check tags
            if not spec.tags or len(spec.tags) == 0:
                return {"valid": False, "error": "No tags specified"}

            return {"valid": True}

        except Exception as e:
            return {"valid": False, "error": str(e)}

    def _run_dry_run_tests(self):
        """Run dry-run workflow tests with mocked dependencies."""
        print("\n🧪 Running Dry-Run Tests")
        print("-" * 25)

        try:
            # Run dry-run validation using pytest
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_e2e_complete_validation.py::TestCompleteE2EValidation::test_specification_compliance_validation",
                "-v",
                "--tb=short",
                "--no-cov",
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=project_root
            )

            self.results["validation_results"]["dry_run"] = {
                "success": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr,
            }

            if result.returncode == 0:
                print("✅ Dry-run tests passed")
            else:
                print("❌ Dry-run tests failed")
                print(f"Error: {result.stderr}")

        except Exception as e:
            self.results["errors"].append(f"Dry-run test failed: {str(e)}")
            print(f"❌ Dry-run test error: {e}")

    def _run_unit_tests(self):
        """Run unit tests."""
        print("\n🔬 Running Unit Tests")
        print("-" * 20)

        try:
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                "tests/",
                "-v",
                "--cov=orchestrator",
                "--cov-report=term-missing",
                "--cov-fail-under=80",
                "-m",
                "not integration and not slow and not performance",
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=project_root
            )

            # Parse coverage from output
            coverage_line = None
            for line in result.stdout.split("\n"):
                if "TOTAL" in line and "%" in line:
                    coverage_line = line
                    break

            coverage_percent = None
            if coverage_line:
                try:
                    coverage_percent = int(coverage_line.split()[-1].replace("%", ""))
                except:
                    pass

            self.results["validation_results"]["unit_tests"] = {
                "success": result.returncode == 0,
                "coverage_percent": coverage_percent,
                "output": result.stdout,
                "errors": result.stderr,
            }

            if result.returncode == 0:
                print(f"✅ Unit tests passed (Coverage: {coverage_percent}%)")
            else:
                print("❌ Unit tests failed")
                print(f"Error: {result.stderr}")

        except Exception as e:
            self.results["errors"].append(f"Unit tests failed: {str(e)}")
            print(f"❌ Unit test error: {e}")

    def _run_integration_tests(self):
        """Run integration tests with mocked MCP servers."""
        print("\n🔗 Running Integration Tests")
        print("-" * 28)

        try:
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_integration_mcp.py",
                "tests/test_e2e_workflow.py::TestE2EWorkflow",
                "-v",
                "-m",
                "integration and not slow",
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=project_root
            )

            self.results["validation_results"]["integration_tests"] = {
                "success": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr,
            }

            if result.returncode == 0:
                print("✅ Integration tests passed")
            else:
                print("❌ Integration tests failed")
                print(f"Error: {result.stderr}")

        except Exception as e:
            self.results["errors"].append(f"Integration tests failed: {str(e)}")
            print(f"❌ Integration test error: {e}")

    def _run_performance_tests(self):
        """Run performance benchmark tests."""
        print("\n⚡ Running Performance Tests")
        print("-" * 28)

        try:
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_performance_benchmarks.py",
                "-v",
                "-m",
                "performance",
                "--durations=10",
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=project_root
            )

            # Parse performance metrics from output
            performance_metrics = self._parse_performance_output(result.stdout)

            self.results["validation_results"]["performance_tests"] = {
                "success": result.returncode == 0,
                "metrics": performance_metrics,
                "output": result.stdout,
                "errors": result.stderr,
            }

            self.results["performance_metrics"] = performance_metrics

            if result.returncode == 0:
                print("✅ Performance tests passed")
                if performance_metrics:
                    print("📊 Performance Metrics:")
                    for metric, value in performance_metrics.items():
                        print(f"   {metric}: {value}")
            else:
                print("❌ Performance tests failed")
                print(f"Error: {result.stderr}")

        except Exception as e:
            self.results["errors"].append(f"Performance tests failed: {str(e)}")
            print(f"❌ Performance test error: {e}")

    def _parse_performance_output(self, output: str) -> Dict[str, Any]:
        """Parse performance metrics from test output."""
        metrics = {}

        try:
            lines = output.split("\n")
            for line in lines:
                # Look for performance output patterns
                if "Duration:" in line:
                    try:
                        duration = float(
                            line.split("Duration:")[1].strip().replace("s", "")
                        )
                        metrics["duration_seconds"] = duration
                    except:
                        pass

                elif "Memory increase:" in line:
                    try:
                        memory = float(
                            line.split("Memory increase:")[1].strip().replace("MB", "")
                        )
                        metrics["memory_increase_mb"] = memory
                    except:
                        pass

                elif "Average CPU:" in line:
                    try:
                        cpu = float(
                            line.split("Average CPU:")[1].strip().replace("%", "")
                        )
                        metrics["avg_cpu_percent"] = cpu
                    except:
                        pass

        except Exception:
            pass

        return metrics

    def _generate_final_report(self):
        """Generate final validation report."""
        print("\n📄 Generating Final Report")
        print("-" * 26)

        # Calculate overall success
        validation_results = self.results["validation_results"]
        total_tests = len(validation_results)
        passed_tests = sum(
            1
            for result in validation_results.values()
            if isinstance(result, dict) and result.get("success", False)
        )

        overall_success = (
            passed_tests == total_tests and len(self.results["errors"]) == 0
        )

        self.results["overall_success"] = overall_success
        self.results["summary"] = {
            "total_test_categories": total_tests,
            "passed_categories": passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "total_errors": len(self.results["errors"]),
        }

        # Save report to file
        report_file = project_root / "e2e_validation_report.json"
        with open(report_file, "w") as f:
            json.dump(self.results, f, indent=2)

        # Print summary
        print(f"📊 Validation Summary:")
        print(f"   Overall Success: {'✅ PASS' if overall_success else '❌ FAIL'}")
        print(f"   Test Categories: {passed_tests}/{total_tests} passed")
        print(f"   Success Rate: {passed_tests/total_tests*100:.1f}%")
        print(f"   Total Errors: {len(self.results['errors'])}")
        print(f"   Report saved: {report_file}")

        if self.results["errors"]:
            print(f"\n❌ Errors encountered:")
            for error in self.results["errors"]:
                print(f"   - {error}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run QA Operator E2E validation")
    parser.add_argument(
        "--skip-specs", action="store_true", help="Skip specification validation"
    )
    parser.add_argument(
        "--skip-dry-run", action="store_true", help="Skip dry-run tests"
    )
    parser.add_argument("--skip-unit", action="store_true", help="Skip unit tests")
    parser.add_argument(
        "--skip-integration", action="store_true", help="Skip integration tests"
    )
    parser.add_argument(
        "--skip-performance", action="store_true", help="Skip performance tests"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file for results (default: e2e_validation_report.json)",
    )

    args = parser.parse_args()

    # Configure validation
    config = {
        "validate_specifications": not args.skip_specs,
        "run_dry_run": not args.skip_dry_run,
        "run_unit_tests": not args.skip_unit,
        "run_integration_tests": not args.skip_integration,
        "run_performance_tests": not args.skip_performance,
        "output_file": args.output,
    }

    # Set environment variables for testing
    os.environ["CI"] = "true"
    os.environ["QA_OPERATOR_LOG_LEVEL"] = "ERROR"
    os.environ["QA_OPERATOR_HEADLESS"] = "true"

    # Run validation
    runner = E2EValidationRunner(config)
    results = runner.run_validation()

    # Exit with appropriate code
    exit_code = 0 if results["overall_success"] else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
