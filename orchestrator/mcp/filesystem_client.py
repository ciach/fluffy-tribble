"""
Filesystem MCP Client Wrapper.

Provides safe file operations with path validation and restriction enforcement
for the Filesystem MCP server, ensuring operations are sandboxed to the e2e/ directory.
"""

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

from ..core.exceptions import FileOperationError, ValidationError, MCPConnectionError


class FilesystemMCPClient:
    """
    Wrapper class for Filesystem MCP tool calls.
    
    Provides safe file operations with path validation and restriction enforcement,
    ensuring all operations are sandboxed to the e2e/ directory for security.
    """

    def __init__(self, connection_manager, e2e_dir: Path, logger: Optional[logging.Logger] = None):
        """
        Initialize the Filesystem MCP client.
        
        Args:
            connection_manager: MCP connection manager instance
            e2e_dir: Path to the e2e directory (sandbox root)
            logger: Optional logger instance
        """
        self.connection_manager = connection_manager
        self.e2e_dir = Path(e2e_dir).resolve()
        self.logger = logger or logging.getLogger(__name__)
        self.server_name = "filesystem"
        
        # Ensure e2e directory exists
        self.e2e_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup directory for file operations
        self.backup_dir = self.e2e_dir / ".backups"
        self.backup_dir.mkdir(exist_ok=True)

    async def _ensure_connection(self) -> Any:
        """Ensure we have a valid connection to the Filesystem MCP server."""
        if not self.connection_manager.is_connected(self.server_name):
            self.logger.warning(f"Filesystem MCP server not connected, attempting to connect...")
            success = await self.connection_manager.connect_server(self.server_name)
            if not success:
                raise MCPConnectionError(
                    f"Failed to connect to Filesystem MCP server",
                    server_name=self.server_name
                )
        
        return self.connection_manager.get_connection(self.server_name)

    def _validate_path(self, path: Union[str, Path]) -> Path:
        """
        Validate that a path is within the e2e directory sandbox.
        
        Args:
            path: Path to validate
            
        Returns:
            Resolved path within sandbox
            
        Raises:
            ValidationError: If path is outside sandbox
        """
        path = Path(path)
        
        # If path is relative, make it relative to e2e_dir
        if not path.is_absolute():
            resolved_path = (self.e2e_dir / path).resolve()
        else:
            resolved_path = path.resolve()
        
        # Check if path is within e2e directory
        try:
            resolved_path.relative_to(self.e2e_dir)
        except ValueError:
            raise ValidationError(
                f"Path outside e2e directory sandbox: {path}",
                validation_type="path_sandbox",
                violations=[f"Attempted access to: {resolved_path}", f"Sandbox root: {self.e2e_dir}"]
            )
        
        return resolved_path

    async def _call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call an MCP tool with error handling.
        
        Args:
            tool_name: Name of the MCP tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            Tool response data
        """
        await self._ensure_connection()
        
        try:
            # TODO: Implement actual MCP tool call
            # This is a placeholder for the actual MCP client call
            self.logger.debug(f"Calling MCP tool: {tool_name} with args: {arguments}")
            
            # Simulate tool call delay
            await asyncio.sleep(0.05)
            
            # Mock response based on tool name
            if tool_name == "read_file":
                file_path = arguments.get("path")
                if file_path and Path(file_path).exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    return {"success": True, "content": content}
                else:
                    return {"success": False, "error": "File not found"}
            
            elif tool_name == "write_file":
                file_path = arguments.get("path")
                content = arguments.get("content", "")
                if file_path:
                    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    return {"success": True, "bytes_written": len(content.encode('utf-8'))}
                else:
                    return {"success": False, "error": "Invalid path"}
            
            elif tool_name == "list_directory":
                dir_path = arguments.get("path")
                if dir_path and Path(dir_path).exists():
                    entries = []
                    for item in Path(dir_path).iterdir():
                        entries.append({
                            "name": item.name,
                            "type": "directory" if item.is_dir() else "file",
                            "size": item.stat().st_size if item.is_file() else None
                        })
                    return {"success": True, "entries": entries}
                else:
                    return {"success": False, "error": "Directory not found"}
            
            elif tool_name == "create_directory":
                dir_path = arguments.get("path")
                if dir_path:
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
                    return {"success": True}
                else:
                    return {"success": False, "error": "Invalid path"}
            
            elif tool_name == "delete_file":
                file_path = arguments.get("path")
                if file_path and Path(file_path).exists():
                    Path(file_path).unlink()
                    return {"success": True}
                else:
                    return {"success": False, "error": "File not found"}
            
            elif tool_name == "copy_file":
                src_path = arguments.get("source")
                dst_path = arguments.get("destination")
                if src_path and dst_path and Path(src_path).exists():
                    shutil.copy2(src_path, dst_path)
                    return {"success": True}
                else:
                    return {"success": False, "error": "Source file not found"}
            
            else:
                return {"success": True}
                
        except Exception as e:
            self.logger.error(f"MCP tool call failed: {tool_name} - {e}")
            raise FileOperationError(
                f"Filesystem MCP tool call failed: {tool_name}",
                file_path=arguments.get("path"),
                operation=tool_name
            )

    async def read_file(self, file_path: Union[str, Path]) -> str:
        """
        Read content from a file.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            File content as string
        """
        validated_path = self._validate_path(file_path)
        
        self.logger.debug(f"Reading file: {validated_path}")
        
        response = await self._call_mcp_tool("read_file", {
            "path": str(validated_path)
        })
        
        if not response.get("success"):
            raise FileOperationError(
                f"Failed to read file: {response.get('error', 'Unknown error')}",
                file_path=str(validated_path),
                operation="read"
            )
        
        return response.get("content", "")

    async def write_file(self, file_path: Union[str, Path], content: str, create_backup: bool = True) -> int:
        """
        Write content to a file.
        
        Args:
            file_path: Path to the file to write
            content: Content to write
            create_backup: Whether to create a backup before writing
            
        Returns:
            Number of bytes written
        """
        validated_path = self._validate_path(file_path)
        
        # Create backup if requested and file exists
        if create_backup and validated_path.exists():
            await self._create_backup(validated_path)
        
        self.logger.debug(f"Writing file: {validated_path}")
        
        response = await self._call_mcp_tool("write_file", {
            "path": str(validated_path),
            "content": content
        })
        
        if not response.get("success"):
            raise FileOperationError(
                f"Failed to write file: {response.get('error', 'Unknown error')}",
                file_path=str(validated_path),
                operation="write"
            )
        
        bytes_written = response.get("bytes_written", len(content.encode('utf-8')))
        self.logger.info(f"Successfully wrote {bytes_written} bytes to {validated_path}")
        return bytes_written

    async def append_file(self, file_path: Union[str, Path], content: str) -> int:
        """
        Append content to a file.
        
        Args:
            file_path: Path to the file to append to
            content: Content to append
            
        Returns:
            Number of bytes appended
        """
        validated_path = self._validate_path(file_path)
        
        # Read existing content if file exists
        existing_content = ""
        if validated_path.exists():
            existing_content = await self.read_file(validated_path)
        
        # Append new content
        new_content = existing_content + content
        return await self.write_file(validated_path, new_content, create_backup=True)

    async def list_directory(self, dir_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        List contents of a directory.
        
        Args:
            dir_path: Path to the directory to list
            
        Returns:
            List of directory entries with metadata
        """
        validated_path = self._validate_path(dir_path)
        
        self.logger.debug(f"Listing directory: {validated_path}")
        
        response = await self._call_mcp_tool("list_directory", {
            "path": str(validated_path)
        })
        
        if not response.get("success"):
            raise FileOperationError(
                f"Failed to list directory: {response.get('error', 'Unknown error')}",
                file_path=str(validated_path),
                operation="list"
            )
        
        return response.get("entries", [])

    async def create_directory(self, dir_path: Union[str, Path], parents: bool = True) -> None:
        """
        Create a directory.
        
        Args:
            dir_path: Path to the directory to create
            parents: Whether to create parent directories
        """
        validated_path = self._validate_path(dir_path)
        
        self.logger.debug(f"Creating directory: {validated_path}")
        
        response = await self._call_mcp_tool("create_directory", {
            "path": str(validated_path),
            "parents": parents
        })
        
        if not response.get("success"):
            raise FileOperationError(
                f"Failed to create directory: {response.get('error', 'Unknown error')}",
                file_path=str(validated_path),
                operation="create_directory"
            )

    async def delete_file(self, file_path: Union[str, Path], create_backup: bool = True) -> None:
        """
        Delete a file.
        
        Args:
            file_path: Path to the file to delete
            create_backup: Whether to create a backup before deletion
        """
        validated_path = self._validate_path(file_path)
        
        # Create backup if requested and file exists
        if create_backup and validated_path.exists():
            await self._create_backup(validated_path)
        
        self.logger.debug(f"Deleting file: {validated_path}")
        
        response = await self._call_mcp_tool("delete_file", {
            "path": str(validated_path)
        })
        
        if not response.get("success"):
            raise FileOperationError(
                f"Failed to delete file: {response.get('error', 'Unknown error')}",
                file_path=str(validated_path),
                operation="delete"
            )

    async def copy_file(self, source_path: Union[str, Path], dest_path: Union[str, Path]) -> None:
        """
        Copy a file.
        
        Args:
            source_path: Path to the source file
            dest_path: Path to the destination file
        """
        validated_source = self._validate_path(source_path)
        validated_dest = self._validate_path(dest_path)
        
        self.logger.debug(f"Copying file: {validated_source} -> {validated_dest}")
        
        response = await self._call_mcp_tool("copy_file", {
            "source": str(validated_source),
            "destination": str(validated_dest)
        })
        
        if not response.get("success"):
            raise FileOperationError(
                f"Failed to copy file: {response.get('error', 'Unknown error')}",
                file_path=str(validated_source),
                operation="copy"
            )

    async def move_file(self, source_path: Union[str, Path], dest_path: Union[str, Path]) -> None:
        """
        Move a file.
        
        Args:
            source_path: Path to the source file
            dest_path: Path to the destination file
        """
        # Copy then delete
        await self.copy_file(source_path, dest_path)
        await self.delete_file(source_path, create_backup=False)  # Don't backup since we copied

    async def file_exists(self, file_path: Union[str, Path]) -> bool:
        """
        Check if a file exists.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file exists, False otherwise
        """
        try:
            validated_path = self._validate_path(file_path)
            return validated_path.exists()
        except ValidationError:
            return False

    async def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get file information.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file metadata
        """
        validated_path = self._validate_path(file_path)
        
        if not validated_path.exists():
            raise FileOperationError(
                f"File not found: {validated_path}",
                file_path=str(validated_path),
                operation="stat"
            )
        
        stat = validated_path.stat()
        
        return {
            "path": str(validated_path),
            "name": validated_path.name,
            "size": stat.st_size,
            "is_file": validated_path.is_file(),
            "is_directory": validated_path.is_dir(),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "permissions": oct(stat.st_mode)[-3:]
        }

    async def _create_backup(self, file_path: Path) -> str:
        """
        Create a backup of a file.
        
        Args:
            file_path: Path to the file to backup
            
        Returns:
            Path to the backup file
        """
        if not file_path.exists():
            return ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.name}.backup_{timestamp}"
        backup_path = self.backup_dir / backup_name
        
        try:
            await self.copy_file(file_path, backup_path)
            self.logger.debug(f"Created backup: {backup_path}")
            return str(backup_path)
        except Exception as e:
            self.logger.warning(f"Failed to create backup for {file_path}: {e}")
            return ""

    async def restore_backup(self, original_path: Union[str, Path], backup_timestamp: Optional[str] = None) -> bool:
        """
        Restore a file from backup.
        
        Args:
            original_path: Path to the original file
            backup_timestamp: Specific backup timestamp, or None for latest
            
        Returns:
            True if restore successful, False otherwise
        """
        validated_path = self._validate_path(original_path)
        file_name = validated_path.name
        
        # Find backup files
        backup_pattern = f"{file_name}.backup_*"
        backup_files = list(self.backup_dir.glob(backup_pattern))
        
        if not backup_files:
            self.logger.warning(f"No backup files found for {file_name}")
            return False
        
        # Select backup file
        if backup_timestamp:
            backup_file = self.backup_dir / f"{file_name}.backup_{backup_timestamp}"
            if not backup_file.exists():
                self.logger.warning(f"Backup file not found: {backup_file}")
                return False
        else:
            # Use latest backup
            backup_file = max(backup_files, key=lambda p: p.stat().st_mtime)
        
        try:
            await self.copy_file(backup_file, validated_path)
            self.logger.info(f"Restored {validated_path} from backup {backup_file}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to restore backup: {e}")
            return False

    async def list_backups(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        List available backups for a file.
        
        Args:
            file_path: Path to the original file
            
        Returns:
            List of backup information
        """
        validated_path = self._validate_path(file_path)
        file_name = validated_path.name
        
        backup_pattern = f"{file_name}.backup_*"
        backup_files = list(self.backup_dir.glob(backup_pattern))
        
        backups = []
        for backup_file in backup_files:
            # Extract timestamp from filename
            timestamp_part = backup_file.name.split('.backup_')[-1]
            stat = backup_file.stat()
            
            backups.append({
                "timestamp": timestamp_part,
                "path": str(backup_file),
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
            })
        
        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x["timestamp"], reverse=True)
        return backups

    async def validate_directory_structure(self) -> Dict[str, Any]:
        """
        Validate the e2e directory structure and create missing directories.
        
        Returns:
            Validation results with created directories
        """
        self.logger.info("Validating e2e directory structure")
        
        # Standard e2e directory structure
        required_dirs = [
            "tests",
            "fixtures",
            "pages",
            "utils",
            "config"
        ]
        
        created_dirs = []
        existing_dirs = []
        
        for dir_name in required_dirs:
            dir_path = self.e2e_dir / dir_name
            if dir_path.exists():
                existing_dirs.append(dir_name)
            else:
                try:
                    await self.create_directory(dir_path)
                    created_dirs.append(dir_name)
                except Exception as e:
                    self.logger.error(f"Failed to create directory {dir_name}: {e}")
        
        # Ensure backup directory exists
        if not self.backup_dir.exists():
            self.backup_dir.mkdir(exist_ok=True)
            created_dirs.append(".backups")
        
        result = {
            "sandbox_root": str(self.e2e_dir),
            "existing_directories": existing_dirs,
            "created_directories": created_dirs,
            "validation_passed": True
        }
        
        self.logger.info(f"Directory structure validation complete: {len(created_dirs)} directories created")
        return result

    def get_sandbox_root(self) -> Path:
        """
        Get the sandbox root directory.
        
        Returns:
            Path to the e2e directory
        """
        return self.e2e_dir

    def is_available(self) -> bool:
        """
        Check if Filesystem MCP server is available.
        
        Returns:
            True if server is connected and available
        """
        return self.connection_manager.is_connected(self.server_name)

    async def cleanup_old_backups(self, max_age_days: int = 7) -> int:
        """
        Clean up old backup files.
        
        Args:
            max_age_days: Maximum age of backups to keep
            
        Returns:
            Number of backup files deleted
        """
        import time
        
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        deleted_count = 0
        
        for backup_file in self.backup_dir.glob("*.backup_*"):
            file_age = current_time - backup_file.stat().st_mtime
            
            if file_age > max_age_seconds:
                try:
                    backup_file.unlink()
                    deleted_count += 1
                    self.logger.debug(f"Deleted old backup: {backup_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to delete backup {backup_file}: {e}")
        
        if deleted_count > 0:
            self.logger.info(f"Cleaned up {deleted_count} old backup files")
        
        return deleted_count