"""
Artifact storage and cleanup system.

Provides organized artifact storage, retention policy enforcement,
and cleanup utilities with scheduled execution.
"""

import asyncio
import gzip
import hashlib
import json
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Union

from ..core.config import Config
from ..core.logging_config import get_logger, log_performance
from .models import ArtifactMetadata, ArtifactType


class ArtifactManager:
    """
    Manages test artifact storage, compression, and cleanup.
    
    Provides organized storage in artifacts/ directory with configurable
    retention policies and automatic cleanup utilities.
    """
    
    def __init__(self, config: Config, workflow_id: str):
        """
        Initialize the artifact manager.
        
        Args:
            config: QA Operator configuration
            workflow_id: Unique workflow identifier
        """
        self.config = config
        self.workflow_id = workflow_id
        self.logger = get_logger(__name__, workflow_id=workflow_id)
        
        # Ensure artifacts directory exists
        self.artifacts_root = config.artifacts_dir
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        
        # Metadata tracking
        self._artifact_registry: Dict[str, ArtifactMetadata] = {}
        self._registry_file = self.artifacts_root / "registry.json"
        
        # Load existing registry
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load artifact registry from disk."""
        if self._registry_file.exists():
            try:
                with open(self._registry_file, "r", encoding="utf-8") as f:
                    registry_data = json.load(f)
                
                for artifact_id, data in registry_data.items():
                    # Convert datetime strings back to datetime objects
                    if "created_at" in data:
                        data["created_at"] = datetime.fromisoformat(data["created_at"])
                    
                    # Convert artifact_type string back to enum
                    if "artifact_type" in data:
                        data["artifact_type"] = ArtifactType(data["artifact_type"])
                    
                    self._artifact_registry[artifact_id] = ArtifactMetadata(**data)
                
                self.logger.debug(f"Loaded {len(self._artifact_registry)} artifacts from registry")
                
            except Exception as e:
                self.logger.warning(f"Failed to load artifact registry: {e}")
                self._artifact_registry = {}
    
    def _save_registry(self) -> None:
        """Save artifact registry to disk."""
        try:
            registry_data = {}
            for artifact_id, metadata in self._artifact_registry.items():
                data = metadata.model_dump()
                # Convert datetime to string for JSON serialization
                data["created_at"] = data["created_at"].isoformat()
                # Convert enum to string
                data["artifact_type"] = data["artifact_type"].value
                registry_data[artifact_id] = data
            
            with open(self._registry_file, "w", encoding="utf-8") as f:
                json.dump(registry_data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Saved {len(self._artifact_registry)} artifacts to registry")
            
        except Exception as e:
            self.logger.error(f"Failed to save artifact registry: {e}")
    
    def _generate_artifact_id(self, metadata: ArtifactMetadata) -> str:
        """Generate unique artifact ID."""
        content = f"{metadata.test_name}_{metadata.artifact_type.value}_{metadata.created_at.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    async def prepare_test_artifacts_dir(self, test_name: str) -> Path:
        """
        Prepare artifacts directory for a test.
        
        Args:
            test_name: Name of the test
            
        Returns:
            Path to the test's artifact directory
        """
        # Create directory structure: artifacts/{timestamp}/{test_name}/
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        test_dir = self.artifacts_root / timestamp / test_name
        test_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.debug(f"Prepared artifact directory: {test_dir}")
        return test_dir
    
    def register_artifact(self, metadata: ArtifactMetadata) -> str:
        """
        Register an artifact in the management system.
        
        Args:
            metadata: Artifact metadata
            
        Returns:
            Unique artifact ID
        """
        artifact_id = self._generate_artifact_id(metadata)
        
        # Calculate checksum if not provided
        if not metadata.checksum and Path(metadata.file_path).exists():
            metadata.checksum = self._calculate_checksum(metadata.file_path)
        
        self._artifact_registry[artifact_id] = metadata
        self._save_registry()
        
        self.logger.debug(
            f"Registered artifact: {artifact_id}",
            extra={
                "metadata": {
                    "artifact_id": artifact_id,
                    "type": metadata.artifact_type.value,
                    "test_name": metadata.test_name,
                    "file_size": metadata.file_size,
                }
            }
        )
        
        return artifact_id
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            self.logger.warning(f"Failed to calculate checksum for {file_path}: {e}")
            return ""
    
    def get_artifacts_by_test(self, test_name: str) -> List[ArtifactMetadata]:
        """
        Get all artifacts for a specific test.
        
        Args:
            test_name: Name of the test
            
        Returns:
            List of artifact metadata for the test
        """
        return [
            metadata for metadata in self._artifact_registry.values()
            if metadata.test_name == test_name
        ]
    
    def get_artifacts_by_type(self, artifact_type: ArtifactType) -> List[ArtifactMetadata]:
        """
        Get all artifacts of a specific type.
        
        Args:
            artifact_type: Type of artifacts to retrieve
            
        Returns:
            List of artifact metadata of the specified type
        """
        return [
            metadata for metadata in self._artifact_registry.values()
            if metadata.artifact_type == artifact_type
        ]
    
    def get_artifacts_by_workflow(self, workflow_id: str) -> List[ArtifactMetadata]:
        """
        Get all artifacts for a specific workflow.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            List of artifact metadata for the workflow
        """
        return [
            metadata for metadata in self._artifact_registry.values()
            if metadata.workflow_id == workflow_id
        ]
    
    def get_expired_artifacts(self, retention_days: Optional[int] = None) -> List[ArtifactMetadata]:
        """
        Get artifacts that have exceeded the retention period.
        
        Args:
            retention_days: Override default retention period
            
        Returns:
            List of expired artifact metadata
        """
        if retention_days is None:
            retention_days = self.config.artifact_retention_days
        
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        expired = [
            metadata for metadata in self._artifact_registry.values()
            if metadata.created_at < cutoff_date
        ]
        
        self.logger.debug(
            f"Found {len(expired)} expired artifacts (retention: {retention_days} days)",
            extra={
                "metadata": {
                    "expired_count": len(expired),
                    "retention_days": retention_days,
                    "cutoff_date": cutoff_date.isoformat(),
                }
            }
        )
        
        return expired
    
    async def compress_artifact(self, artifact_id: str) -> Optional[str]:
        """
        Compress an artifact to save disk space.
        
        Args:
            artifact_id: ID of the artifact to compress
            
        Returns:
            Path to compressed file, or None if compression failed
        """
        if artifact_id not in self._artifact_registry:
            self.logger.warning(f"Artifact not found for compression: {artifact_id}")
            return None
        
        metadata = self._artifact_registry[artifact_id]
        original_path = Path(metadata.file_path)
        
        if not original_path.exists():
            self.logger.warning(f"Artifact file not found: {original_path}")
            return None
        
        # Skip compression for already compressed files
        if original_path.suffix in [".gz", ".zip", ".7z"]:
            self.logger.debug(f"Skipping compression for already compressed file: {original_path}")
            return str(original_path)
        
        compressed_path = original_path.with_suffix(original_path.suffix + ".gz")
        
        try:
            start_time = time.time()
            
            with open(original_path, "rb") as f_in:
                with gzip.open(compressed_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Update metadata
            original_size = metadata.file_size
            compressed_size = compressed_path.stat().st_size
            compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
            
            # Update registry
            metadata.file_path = str(compressed_path)
            metadata.file_size = compressed_size
            metadata.description = f"{metadata.description} (compressed)"
            self._save_registry()
            
            # Remove original file
            original_path.unlink()
            
            duration = time.time() - start_time
            
            self.logger.info(
                f"Compressed artifact: {artifact_id}",
                extra={
                    "metadata": {
                        "artifact_id": artifact_id,
                        "original_size": original_size,
                        "compressed_size": compressed_size,
                        "compression_ratio": f"{compression_ratio:.2%}",
                        "duration": duration,
                    }
                }
            )
            
            return str(compressed_path)
            
        except Exception as e:
            self.logger.error(f"Failed to compress artifact {artifact_id}: {e}")
            # Clean up partial compressed file
            if compressed_path.exists():
                compressed_path.unlink()
            return None
    
    async def cleanup_expired_artifacts(
        self, 
        retention_days: Optional[int] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Clean up artifacts that have exceeded the retention period.
        
        Args:
            retention_days: Override default retention period
            dry_run: If True, only report what would be deleted
            
        Returns:
            Cleanup summary with statistics
        """
        start_time = time.time()
        
        expired_artifacts = self.get_expired_artifacts(retention_days)
        
        if not expired_artifacts:
            self.logger.info("No expired artifacts found for cleanup")
            return {
                "deleted_count": 0,
                "freed_space": 0,
                "duration": time.time() - start_time,
                "dry_run": dry_run,
            }
        
        deleted_count = 0
        freed_space = 0
        errors = []
        
        for metadata in expired_artifacts:
            try:
                file_path = Path(metadata.file_path)
                file_size = metadata.file_size
                
                if dry_run:
                    self.logger.info(f"Would delete: {file_path} ({file_size} bytes)")
                    deleted_count += 1
                    freed_space += file_size
                else:
                    if file_path.exists():
                        file_path.unlink()
                        deleted_count += 1
                        freed_space += file_size
                        
                        self.logger.debug(f"Deleted expired artifact: {file_path}")
                    
                    # Remove from registry
                    artifact_id = self._generate_artifact_id(metadata)
                    if artifact_id in self._artifact_registry:
                        del self._artifact_registry[artifact_id]
                
            except Exception as e:
                error_msg = f"Failed to delete {metadata.file_path}: {e}"
                errors.append(error_msg)
                self.logger.error(error_msg)
        
        # Save updated registry if not dry run
        if not dry_run:
            self._save_registry()
        
        duration = time.time() - start_time
        
        summary = {
            "deleted_count": deleted_count,
            "freed_space": freed_space,
            "duration": duration,
            "dry_run": dry_run,
            "errors": errors,
        }
        
        self.logger.info(
            f"Artifact cleanup completed: {deleted_count} artifacts, {freed_space} bytes freed",
            extra={
                "metadata": {
                    **summary,
                    "retention_days": retention_days or self.config.artifact_retention_days,
                }
            }
        )
        
        log_performance(
            self.logger,
            "artifact_cleanup",
            duration,
            deleted_count=deleted_count,
            freed_space=freed_space,
        )
        
        return summary
    
    async def cleanup_empty_directories(self) -> int:
        """
        Clean up empty directories in the artifacts tree.
        
        Returns:
            Number of directories removed
        """
        removed_count = 0
        
        # Walk the directory tree bottom-up
        for root, dirs, files in self.artifacts_root.walk(top_down=False):
            for dir_name in dirs:
                dir_path = root / dir_name
                try:
                    # Try to remove if empty
                    if not any(dir_path.iterdir()):
                        dir_path.rmdir()
                        removed_count += 1
                        self.logger.debug(f"Removed empty directory: {dir_path}")
                except OSError:
                    # Directory not empty or other error
                    pass
        
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} empty directories")
        
        return removed_count
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics for artifacts.
        
        Returns:
            Dictionary with storage statistics
        """
        total_artifacts = len(self._artifact_registry)
        total_size = sum(metadata.file_size for metadata in self._artifact_registry.values())
        
        # Group by type
        type_stats = {}
        for metadata in self._artifact_registry.values():
            artifact_type = metadata.artifact_type.value
            if artifact_type not in type_stats:
                type_stats[artifact_type] = {"count": 0, "size": 0}
            type_stats[artifact_type]["count"] += 1
            type_stats[artifact_type]["size"] += metadata.file_size
        
        # Group by age
        now = datetime.utcnow()
        age_stats = {
            "last_24h": 0,
            "last_week": 0,
            "last_month": 0,
            "older": 0,
        }
        
        for metadata in self._artifact_registry.values():
            age = now - metadata.created_at
            if age.days == 0:
                age_stats["last_24h"] += 1
            elif age.days <= 7:
                age_stats["last_week"] += 1
            elif age.days <= 30:
                age_stats["last_month"] += 1
            else:
                age_stats["older"] += 1
        
        return {
            "total_artifacts": total_artifacts,
            "total_size": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "by_type": type_stats,
            "by_age": age_stats,
            "retention_days": self.config.artifact_retention_days,
            "artifacts_root": str(self.artifacts_root),
        }
    
    async def schedule_cleanup(self, interval_hours: int = 24) -> None:
        """
        Schedule periodic artifact cleanup.
        
        Args:
            interval_hours: Cleanup interval in hours
        """
        self.logger.info(f"Starting scheduled cleanup every {interval_hours} hours")
        
        while True:
            try:
                await asyncio.sleep(interval_hours * 3600)  # Convert to seconds
                
                self.logger.info("Running scheduled artifact cleanup")
                
                # Run cleanup
                summary = await self.cleanup_expired_artifacts()
                
                # Clean up empty directories
                empty_dirs = await self.cleanup_empty_directories()
                
                self.logger.info(
                    f"Scheduled cleanup completed: {summary['deleted_count']} artifacts, {empty_dirs} empty directories",
                    extra={"metadata": summary}
                )
                
            except asyncio.CancelledError:
                self.logger.info("Scheduled cleanup cancelled")
                break
            except Exception as e:
                self.logger.error(f"Scheduled cleanup failed: {e}")
                # Continue running despite errors
    
    def validate_artifact_integrity(self, artifact_id: str) -> bool:
        """
        Validate artifact file integrity using checksum.
        
        Args:
            artifact_id: ID of the artifact to validate
            
        Returns:
            True if artifact is valid, False otherwise
        """
        if artifact_id not in self._artifact_registry:
            return False
        
        metadata = self._artifact_registry[artifact_id]
        file_path = Path(metadata.file_path)
        
        if not file_path.exists():
            return False
        
        # Check file size
        actual_size = file_path.stat().st_size
        if actual_size != metadata.file_size:
            self.logger.warning(
                f"Artifact size mismatch: {artifact_id} (expected: {metadata.file_size}, actual: {actual_size})"
            )
            return False
        
        # Check checksum if available
        if metadata.checksum:
            actual_checksum = self._calculate_checksum(str(file_path))
            if actual_checksum != metadata.checksum:
                self.logger.warning(
                    f"Artifact checksum mismatch: {artifact_id}"
                )
                return False
        
        return True
    
    async def run_integrity_check(self) -> Dict[str, Any]:
        """
        Run integrity check on all registered artifacts.
        
        Returns:
            Integrity check results
        """
        start_time = time.time()
        
        total_artifacts = len(self._artifact_registry)
        valid_artifacts = 0
        invalid_artifacts = []
        
        for artifact_id in self._artifact_registry:
            if self.validate_artifact_integrity(artifact_id):
                valid_artifacts += 1
            else:
                invalid_artifacts.append(artifact_id)
        
        duration = time.time() - start_time
        
        results = {
            "total_artifacts": total_artifacts,
            "valid_artifacts": valid_artifacts,
            "invalid_artifacts": len(invalid_artifacts),
            "invalid_artifact_ids": invalid_artifacts,
            "duration": duration,
        }
        
        self.logger.info(
            f"Integrity check completed: {valid_artifacts}/{total_artifacts} valid",
            extra={"metadata": results}
        )
        
        return results