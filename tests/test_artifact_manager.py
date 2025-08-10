"""
Unit tests for artifact manager and retention policy enforcement.

Tests artifact storage, cleanup, and retention policy functionality.
"""

import asyncio
import json
import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from orchestrator.core.config import Config
from orchestrator.execution.artifacts import ArtifactManager
from orchestrator.execution.models import ArtifactMetadata, ArtifactType


@pytest.fixture
def temp_config():
    """Create a temporary configuration for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        config = Config()
        config.artifacts_dir = temp_path / "artifacts"
        config.logs_dir = temp_path / "logs"
        config.artifact_retention_days = 7  # Default for testing
        
        # Create directories
        config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        config.logs_dir.mkdir(parents=True, exist_ok=True)
        
        yield config


@pytest.fixture
def artifact_manager(temp_config):
    """Create an artifact manager instance for testing."""
    return ArtifactManager(temp_config, "test_workflow_123")


@pytest.fixture
def sample_artifact_file(temp_config):
    """Create a sample artifact file for testing."""
    artifacts_dir = temp_config.artifacts_dir
    test_file = artifacts_dir / "sample_artifact.txt"
    test_file.write_text("Sample artifact content for testing")
    return test_file


class TestArtifactManager:
    """Test cases for ArtifactManager."""
    
    def test_initialization(self, artifact_manager, temp_config):
        """Test artifact manager initialization."""
        assert artifact_manager.config == temp_config
        assert artifact_manager.workflow_id == "test_workflow_123"
        assert artifact_manager.artifacts_root == temp_config.artifacts_dir
        assert artifact_manager.artifacts_root.exists()
        assert isinstance(artifact_manager._artifact_registry, dict)
    
    async def test_prepare_test_artifacts_dir(self, artifact_manager):
        """Test preparation of test artifact directories."""
        test_name = "sample_test"
        
        artifacts_dir = await artifact_manager.prepare_test_artifacts_dir(test_name)
        
        assert artifacts_dir.exists()
        assert artifacts_dir.is_dir()
        assert test_name in str(artifacts_dir)
        assert artifacts_dir.parent.parent == artifact_manager.artifacts_root
    
    def test_register_artifact(self, artifact_manager, sample_artifact_file):
        """Test artifact registration."""
        metadata = ArtifactMetadata(
            artifact_type=ArtifactType.SCREENSHOT,
            file_path=str(sample_artifact_file),
            file_size=sample_artifact_file.stat().st_size,
            test_name="test_sample",
            workflow_id="test_workflow_123",
            description="Test screenshot",
        )
        
        artifact_id = artifact_manager.register_artifact(metadata)
        
        assert artifact_id is not None
        assert len(artifact_id) == 16  # MD5 hash truncated to 16 chars
        assert artifact_id in artifact_manager._artifact_registry
        assert artifact_manager._artifact_registry[artifact_id] == metadata
        
        # Check that registry file was created
        registry_file = artifact_manager.artifacts_root / "registry.json"
        assert registry_file.exists()
    
    def test_get_artifacts_by_test(self, artifact_manager, sample_artifact_file):
        """Test retrieving artifacts by test name."""
        # Register multiple artifacts for different tests
        metadata1 = ArtifactMetadata(
            artifact_type=ArtifactType.SCREENSHOT,
            file_path=str(sample_artifact_file),
            file_size=sample_artifact_file.stat().st_size,
            test_name="test_one",
            workflow_id="test_workflow_123",
        )
        
        metadata2 = ArtifactMetadata(
            artifact_type=ArtifactType.TRACE,
            file_path=str(sample_artifact_file),
            file_size=sample_artifact_file.stat().st_size,
            test_name="test_one",
            workflow_id="test_workflow_123",
        )
        
        metadata3 = ArtifactMetadata(
            artifact_type=ArtifactType.VIDEO,
            file_path=str(sample_artifact_file),
            file_size=sample_artifact_file.stat().st_size,
            test_name="test_two",
            workflow_id="test_workflow_123",
        )
        
        artifact_manager.register_artifact(metadata1)
        artifact_manager.register_artifact(metadata2)
        artifact_manager.register_artifact(metadata3)
        
        # Test retrieval
        test_one_artifacts = artifact_manager.get_artifacts_by_test("test_one")
        test_two_artifacts = artifact_manager.get_artifacts_by_test("test_two")
        
        assert len(test_one_artifacts) == 2
        assert len(test_two_artifacts) == 1
        assert all(a.test_name == "test_one" for a in test_one_artifacts)
        assert all(a.test_name == "test_two" for a in test_two_artifacts)
    
    def test_get_artifacts_by_type(self, artifact_manager, sample_artifact_file):
        """Test retrieving artifacts by type."""
        # Register artifacts of different types
        screenshot_metadata = ArtifactMetadata(
            artifact_type=ArtifactType.SCREENSHOT,
            file_path=str(sample_artifact_file),
            file_size=sample_artifact_file.stat().st_size,
            test_name="test_sample",
            workflow_id="test_workflow_123",
        )
        
        trace_metadata = ArtifactMetadata(
            artifact_type=ArtifactType.TRACE,
            file_path=str(sample_artifact_file),
            file_size=sample_artifact_file.stat().st_size,
            test_name="test_sample",
            workflow_id="test_workflow_123",
        )
        
        artifact_manager.register_artifact(screenshot_metadata)
        artifact_manager.register_artifact(trace_metadata)
        
        # Test retrieval by type
        screenshots = artifact_manager.get_artifacts_by_type(ArtifactType.SCREENSHOT)
        traces = artifact_manager.get_artifacts_by_type(ArtifactType.TRACE)
        videos = artifact_manager.get_artifacts_by_type(ArtifactType.VIDEO)
        
        assert len(screenshots) == 1
        assert len(traces) == 1
        assert len(videos) == 0
        assert screenshots[0].artifact_type == ArtifactType.SCREENSHOT
        assert traces[0].artifact_type == ArtifactType.TRACE
    
    def test_get_expired_artifacts_default_retention(self, artifact_manager, sample_artifact_file):
        """Test getting expired artifacts with default retention policy (7 days)."""
        # Create artifacts with different ages
        now = datetime.utcnow()
        
        # Recent artifact (should not be expired)
        recent_metadata = ArtifactMetadata(
            artifact_type=ArtifactType.SCREENSHOT,
            file_path=str(sample_artifact_file),
            file_size=sample_artifact_file.stat().st_size,
            test_name="recent_test",
            workflow_id="test_workflow_123",
            created_at=now - timedelta(days=3),
        )
        
        # Old artifact (should be expired with 7-day retention)
        old_metadata = ArtifactMetadata(
            artifact_type=ArtifactType.TRACE,
            file_path=str(sample_artifact_file),
            file_size=sample_artifact_file.stat().st_size,
            test_name="old_test",
            workflow_id="test_workflow_123",
            created_at=now - timedelta(days=10),
        )
        
        artifact_manager.register_artifact(recent_metadata)
        artifact_manager.register_artifact(old_metadata)
        
        # Test with default retention (7 days)
        expired_artifacts = artifact_manager.get_expired_artifacts()
        
        assert len(expired_artifacts) == 1
        assert expired_artifacts[0].test_name == "old_test"
        assert expired_artifacts[0].artifact_type == ArtifactType.TRACE
    
    def test_get_expired_artifacts_ci_retention(self, artifact_manager, sample_artifact_file):
        """Test getting expired artifacts with CI retention policy (30 days)."""
        # Update config to simulate CI environment
        artifact_manager.config.artifact_retention_days = 30
        
        now = datetime.utcnow()
        
        # Artifact that would be expired with 7-day retention but not 30-day
        medium_age_metadata = ArtifactMetadata(
            artifact_type=ArtifactType.SCREENSHOT,
            file_path=str(sample_artifact_file),
            file_size=sample_artifact_file.stat().st_size,
            test_name="medium_age_test",
            workflow_id="test_workflow_123",
            created_at=now - timedelta(days=15),
        )
        
        # Very old artifact (should be expired even with 30-day retention)
        very_old_metadata = ArtifactMetadata(
            artifact_type=ArtifactType.TRACE,
            file_path=str(sample_artifact_file),
            file_size=sample_artifact_file.stat().st_size,
            test_name="very_old_test",
            workflow_id="test_workflow_123",
            created_at=now - timedelta(days=35),
        )
        
        artifact_manager.register_artifact(medium_age_metadata)
        artifact_manager.register_artifact(very_old_metadata)
        
        # Test with 30-day retention
        expired_artifacts = artifact_manager.get_expired_artifacts()
        
        assert len(expired_artifacts) == 1
        assert expired_artifacts[0].test_name == "very_old_test"
    
    def test_get_expired_artifacts_custom_retention(self, artifact_manager, sample_artifact_file):
        """Test getting expired artifacts with custom retention override."""
        now = datetime.utcnow()
        
        # Create artifact that's 5 days old
        metadata = ArtifactMetadata(
            artifact_type=ArtifactType.SCREENSHOT,
            file_path=str(sample_artifact_file),
            file_size=sample_artifact_file.stat().st_size,
            test_name="test_sample",
            workflow_id="test_workflow_123",
            created_at=now - timedelta(days=5),
        )
        
        artifact_manager.register_artifact(metadata)
        
        # Test with different retention periods
        expired_3_days = artifact_manager.get_expired_artifacts(retention_days=3)
        expired_7_days = artifact_manager.get_expired_artifacts(retention_days=7)
        
        assert len(expired_3_days) == 1  # 5 days > 3 days
        assert len(expired_7_days) == 0  # 5 days < 7 days
    
    def test_retention_policy_environment_variable_override(self, temp_config):
        """Test retention policy override via environment variable."""
        # Test with environment variable override
        with patch.dict(os.environ, {"QA_OPERATOR_ARTIFACT_RETENTION_DAYS": "14"}):
            config = Config()
            config.artifacts_dir = temp_config.artifacts_dir
            config.logs_dir = temp_config.logs_dir
            
            # Reinitialize config to pick up environment variable
            config.artifact_retention_days = int(
                os.getenv("QA_OPERATOR_ARTIFACT_RETENTION_DAYS", "7")
            )
            
            manager = ArtifactManager(config, "test_workflow")
            
            assert manager.config.artifact_retention_days == 14
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_artifacts_dry_run(self, artifact_manager, sample_artifact_file):
        """Test dry run cleanup of expired artifacts."""
        now = datetime.utcnow()
        
        # Create expired artifact
        expired_metadata = ArtifactMetadata(
            artifact_type=ArtifactType.SCREENSHOT,
            file_path=str(sample_artifact_file),
            file_size=sample_artifact_file.stat().st_size,
            test_name="expired_test",
            workflow_id="test_workflow_123",
            created_at=now - timedelta(days=10),
        )
        
        artifact_id = artifact_manager.register_artifact(expired_metadata)
        
        # Run dry run cleanup
        summary = await artifact_manager.cleanup_expired_artifacts(dry_run=True)
        
        # Verify dry run results
        assert summary["dry_run"] is True
        assert summary["deleted_count"] == 1
        assert summary["freed_space"] == sample_artifact_file.stat().st_size
        assert len(summary["errors"]) == 0
        
        # Verify file still exists (dry run)
        assert sample_artifact_file.exists()
        assert artifact_id in artifact_manager._artifact_registry
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_artifacts_actual(self, artifact_manager, temp_config):
        """Test actual cleanup of expired artifacts."""
        # Create a temporary artifact file that we can delete
        expired_file = temp_config.artifacts_dir / "expired_artifact.txt"
        expired_file.write_text("Expired content")
        
        now = datetime.utcnow()
        
        # Create expired artifact
        expired_metadata = ArtifactMetadata(
            artifact_type=ArtifactType.SCREENSHOT,
            file_path=str(expired_file),
            file_size=expired_file.stat().st_size,
            test_name="expired_test",
            workflow_id="test_workflow_123",
            created_at=now - timedelta(days=10),
        )
        
        artifact_id = artifact_manager.register_artifact(expired_metadata)
        
        # Run actual cleanup
        summary = await artifact_manager.cleanup_expired_artifacts(dry_run=False)
        
        # Verify cleanup results
        assert summary["dry_run"] is False
        assert summary["deleted_count"] == 1
        assert summary["freed_space"] > 0
        assert len(summary["errors"]) == 0
        
        # Verify file was deleted
        assert not expired_file.exists()
        assert artifact_id not in artifact_manager._artifact_registry
    
    @pytest.mark.asyncio
    async def test_cleanup_no_expired_artifacts(self, artifact_manager, sample_artifact_file):
        """Test cleanup when no artifacts are expired."""
        # Create recent artifact
        recent_metadata = ArtifactMetadata(
            artifact_type=ArtifactType.SCREENSHOT,
            file_path=str(sample_artifact_file),
            file_size=sample_artifact_file.stat().st_size,
            test_name="recent_test",
            workflow_id="test_workflow_123",
            created_at=datetime.utcnow() - timedelta(days=1),
        )
        
        artifact_manager.register_artifact(recent_metadata)
        
        # Run cleanup
        summary = await artifact_manager.cleanup_expired_artifacts()
        
        # Verify no cleanup occurred
        assert summary["deleted_count"] == 0
        assert summary["freed_space"] == 0
        assert len(summary["errors"]) == 0
    
    def test_storage_statistics(self, artifact_manager, sample_artifact_file):
        """Test storage statistics calculation."""
        # Register artifacts of different types and ages
        now = datetime.utcnow()
        
        artifacts = [
            ArtifactMetadata(
                artifact_type=ArtifactType.SCREENSHOT,
                file_path=str(sample_artifact_file),
                file_size=1000,
                test_name="test1",
                workflow_id="test_workflow_123",
                created_at=now - timedelta(hours=12),  # Last 24h
            ),
            ArtifactMetadata(
                artifact_type=ArtifactType.SCREENSHOT,
                file_path=str(sample_artifact_file),
                file_size=2000,
                test_name="test2",
                workflow_id="test_workflow_123",
                created_at=now - timedelta(days=3),  # Last week
            ),
            ArtifactMetadata(
                artifact_type=ArtifactType.TRACE,
                file_path=str(sample_artifact_file),
                file_size=5000,
                test_name="test3",
                workflow_id="test_workflow_123",
                created_at=now - timedelta(days=15),  # Last month
            ),
            ArtifactMetadata(
                artifact_type=ArtifactType.VIDEO,
                file_path=str(sample_artifact_file),
                file_size=10000,
                test_name="test4",
                workflow_id="test_workflow_123",
                created_at=now - timedelta(days=45),  # Older
            ),
        ]
        
        for artifact in artifacts:
            artifact_manager.register_artifact(artifact)
        
        # Get statistics
        stats = artifact_manager.get_storage_statistics()
        
        # Verify statistics
        assert stats["total_artifacts"] == 4
        assert stats["total_size"] == 18000  # 1000 + 2000 + 5000 + 10000
        assert stats["total_size_mb"] == 18000 / (1024 * 1024)
        
        # Verify type statistics
        assert stats["by_type"]["screenshot"]["count"] == 2
        assert stats["by_type"]["screenshot"]["size"] == 3000
        assert stats["by_type"]["trace"]["count"] == 1
        assert stats["by_type"]["trace"]["size"] == 5000
        assert stats["by_type"]["video"]["count"] == 1
        assert stats["by_type"]["video"]["size"] == 10000
        
        # Verify age statistics
        assert stats["by_age"]["last_24h"] == 1
        assert stats["by_age"]["last_week"] == 1
        assert stats["by_age"]["last_month"] == 1
        assert stats["by_age"]["older"] == 1
        
        assert stats["retention_days"] == artifact_manager.config.artifact_retention_days
    
    def test_registry_persistence(self, temp_config):
        """Test that artifact registry persists across manager instances."""
        # Create first manager and register artifact
        manager1 = ArtifactManager(temp_config, "test_workflow_123")
        
        # Create a test file
        test_file = temp_config.artifacts_dir / "test_artifact.txt"
        test_file.write_text("Test content")
        
        metadata = ArtifactMetadata(
            artifact_type=ArtifactType.SCREENSHOT,
            file_path=str(test_file),
            file_size=test_file.stat().st_size,
            test_name="test_sample",
            workflow_id="test_workflow_123",
        )
        
        artifact_id = manager1.register_artifact(metadata)
        
        # Create second manager instance
        manager2 = ArtifactManager(temp_config, "test_workflow_456")
        
        # Verify artifact was loaded from registry
        assert artifact_id in manager2._artifact_registry
        assert manager2._artifact_registry[artifact_id].test_name == "test_sample"
        assert manager2._artifact_registry[artifact_id].artifact_type == ArtifactType.SCREENSHOT


@pytest.mark.asyncio
class TestArtifactManagerAsync:
    """Async test cases for ArtifactManager."""
    
    async def test_compress_artifact(self, artifact_manager, temp_config):
        """Test artifact compression functionality."""
        # Create a larger test file for compression
        test_file = temp_config.artifacts_dir / "large_artifact.txt"
        test_content = "This is test content for compression. " * 100
        test_file.write_text(test_content)
        
        metadata = ArtifactMetadata(
            artifact_type=ArtifactType.CONSOLE_LOG,
            file_path=str(test_file),
            file_size=test_file.stat().st_size,
            test_name="compression_test",
            workflow_id="test_workflow_123",
        )
        
        artifact_id = artifact_manager.register_artifact(metadata)
        original_size = test_file.stat().st_size
        
        # Compress the artifact
        compressed_path = await artifact_manager.compress_artifact(artifact_id)
        
        assert compressed_path is not None
        assert compressed_path.endswith(".gz")
        assert Path(compressed_path).exists()
        assert not test_file.exists()  # Original should be deleted
        
        # Verify metadata was updated
        updated_metadata = artifact_manager._artifact_registry[artifact_id]
        assert updated_metadata.file_path == compressed_path
        assert updated_metadata.file_size < original_size  # Should be smaller
        assert "compressed" in updated_metadata.description
    
    async def test_cleanup_empty_directories(self, artifact_manager, temp_config):
        """Test cleanup of empty directories."""
        # Create nested empty directories
        empty_dir1 = temp_config.artifacts_dir / "empty1"
        empty_dir2 = temp_config.artifacts_dir / "nested" / "empty2"
        empty_dir1.mkdir()
        empty_dir2.mkdir(parents=True)
        
        # Create directory with file (should not be removed)
        dir_with_file = temp_config.artifacts_dir / "with_file"
        dir_with_file.mkdir()
        (dir_with_file / "file.txt").write_text("content")
        
        # Run cleanup
        removed_count = await artifact_manager.cleanup_empty_directories()
        
        # Verify empty directories were removed
        assert removed_count >= 2  # At least empty1 and empty2
        assert not empty_dir1.exists()
        assert not empty_dir2.exists()
        
        # Verify directory with file was not removed
        assert dir_with_file.exists()
        assert (dir_with_file / "file.txt").exists()
    
    async def test_integrity_check(self, artifact_manager, temp_config):
        """Test artifact integrity checking."""
        # Create valid artifact
        valid_file = temp_config.artifacts_dir / "valid_artifact.txt"
        valid_file.write_text("Valid content")
        
        valid_metadata = ArtifactMetadata(
            artifact_type=ArtifactType.SCREENSHOT,
            file_path=str(valid_file),
            file_size=valid_file.stat().st_size,
            test_name="valid_test",
            workflow_id="test_workflow_123",
        )
        
        valid_id = artifact_manager.register_artifact(valid_metadata)
        
        # Create invalid artifact (file doesn't exist)
        invalid_metadata = ArtifactMetadata(
            artifact_type=ArtifactType.TRACE,
            file_path=str(temp_config.artifacts_dir / "nonexistent.txt"),
            file_size=1000,
            test_name="invalid_test",
            workflow_id="test_workflow_123",
        )
        
        invalid_id = artifact_manager.register_artifact(invalid_metadata)
        
        # Run integrity check
        results = await artifact_manager.run_integrity_check()
        
        # Verify results
        assert results["total_artifacts"] == 2
        assert results["valid_artifacts"] == 1
        assert results["invalid_artifacts"] == 1
        assert invalid_id in results["invalid_artifact_ids"]
        assert valid_id not in results["invalid_artifact_ids"]


if __name__ == "__main__":
    pytest.main([__file__])