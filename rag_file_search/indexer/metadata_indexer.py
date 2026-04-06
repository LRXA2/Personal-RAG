"""
Metadata indexer for scanning and indexing file metadata.

This module handles:
- Scanning directories recursively
- Extracting metadata from files
- Building an in-memory index for fast retrieval
- Fuzzy matching on filenames and paths
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import fnmatch
import pickle
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import FileMetadata
from core.policy import SafetyPolicy


class MetadataIndexer:
    """Indexes file metadata for fast retrieval."""
    
    def __init__(self, policy: Optional[SafetyPolicy] = None):
        self.policy = policy or SafetyPolicy()
        self.index: list[FileMetadata] = []
        self._path_to_metadata: dict[str, FileMetadata] = {}
    
    def scan_directory(self, directory: str, progress_callback=None) -> int:
        """
        Scan a directory and index all files.
        
        Args:
            directory: Root directory to scan
            progress_callback: Optional callback(current_file_count, current_path)
        
        Returns:
            Number of files indexed
        """
        root_path = Path(directory).resolve()
        count = 0
        
        for dirpath, dirnames, filenames in os.walk(root_path, followlinks=False):
            # Index the current directory itself
            dir_metadata = None
            dir_key = None
            try:
                current_dir = Path(dirpath)
                allowed, _ = self.policy.is_path_allowed(str(current_dir))
                if allowed:
                    dir_key = str(current_dir.resolve()).lower()
                    if dir_key not in self._path_to_metadata:
                        dir_stat = current_dir.stat()
                        display_name = current_dir.name or current_dir.drive or str(current_dir)
                        dir_metadata = FileMetadata(
                            filename=display_name,
                            full_path=str(current_dir.resolve()),
                            parent_folder=str(current_dir.resolve().parent),
                            extension="",
                            file_type="folder",
                            size_bytes=0,
                            modified_date=datetime.fromtimestamp(dir_stat.st_mtime),
                            created_date=datetime.fromtimestamp(dir_stat.st_ctime),
                        )
                        self.index.append(dir_metadata)
                        self._path_to_metadata[dir_key] = dir_metadata
                        count += 1

                        if progress_callback:
                            progress_callback(count, str(current_dir))
                    else:
                        dir_metadata = self._path_to_metadata.get(dir_key)
            except (OSError, PermissionError):
                pass

            # Filter out blocked directories
            dirnames[:] = [
                d for d in dirnames
                if d.lower() not in self.policy.blocked_dirs
            ]
            
            dir_size_bytes = 0
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                
                try:
                    # Check if path is allowed
                    allowed, _ = self.policy.is_path_allowed(file_path)
                    if not allowed:
                        continue
                    
                    # Get file stats
                    stat = os.stat(file_path)
                    
                    # Create metadata
                    metadata = FileMetadata(
                        filename=filename,
                        full_path=os.path.abspath(file_path),
                        parent_folder=os.path.dirname(os.path.abspath(file_path)),
                        extension=Path(filename).suffix.lower(),
                        file_type=self.policy.get_file_type(file_path),
                        size_bytes=stat.st_size,
                        modified_date=datetime.fromtimestamp(stat.st_mtime),
                        created_date=datetime.fromtimestamp(stat.st_ctime),
                    )
                    
                    self.index.append(metadata)
                    self._path_to_metadata[metadata.full_path.lower()] = metadata
                    count += 1
                    dir_size_bytes += stat.st_size
                    
                    if progress_callback:
                        progress_callback(count, file_path)
                        
                except (OSError, PermissionError):
                    # Skip files we can't access
                    continue

            if dir_metadata is not None:
                dir_metadata.size_bytes = dir_size_bytes
        
        return count

    def refresh_directory(self, directory: str, progress_callback=None) -> dict:
        """
        Incrementally refresh a directory against existing index state.

        Adds new entries, updates changed ones, and removes deleted paths.

        Returns:
            Dict with counts: scanned, added, updated, removed
        """
        root_path = Path(directory).resolve()
        root_key = str(root_path).lower()

        existing_under_root = {
            p for p in self._path_to_metadata.keys()
            if p == root_key or p.startswith(root_key + os.sep)
        }

        seen_paths: set[str] = set()
        scanned = 0
        added = 0
        updated = 0

        for dirpath, dirnames, filenames in os.walk(root_path, followlinks=False):
            # Filter blocked directories before descending
            dirnames[:] = [
                d for d in dirnames
                if d.lower() not in self.policy.blocked_dirs
            ]

            current_dir = Path(dirpath)
            try:
                allowed, _ = self.policy.is_path_allowed(str(current_dir))
                if not allowed:
                    continue

                dir_resolved = current_dir.resolve()
                dir_key = str(dir_resolved).lower()
                seen_paths.add(dir_key)

                dir_stat = current_dir.stat()
                display_name = current_dir.name or current_dir.drive or str(current_dir)
                direct_file_bytes = 0

                existing_dir = self._path_to_metadata.get(dir_key)
                if existing_dir is None:
                    dir_metadata = FileMetadata(
                        filename=display_name,
                        full_path=str(dir_resolved),
                        parent_folder=str(dir_resolved.parent),
                        extension="",
                        file_type="folder",
                        size_bytes=0,
                        modified_date=datetime.fromtimestamp(dir_stat.st_mtime),
                        created_date=datetime.fromtimestamp(dir_stat.st_ctime),
                    )
                    self.index.append(dir_metadata)
                    self._path_to_metadata[dir_key] = dir_metadata
                    added += 1
                    scanned += 1
                    if progress_callback:
                        progress_callback(scanned, str(current_dir))
                    existing_dir = dir_metadata
                else:
                    old_modified = existing_dir.modified_date
                    old_created = existing_dir.created_date
                    existing_dir.filename = display_name
                    existing_dir.parent_folder = str(dir_resolved.parent)
                    existing_dir.file_type = "folder"
                    existing_dir.extension = ""
                    existing_dir.modified_date = datetime.fromtimestamp(dir_stat.st_mtime)
                    existing_dir.created_date = datetime.fromtimestamp(dir_stat.st_ctime)
                    if old_modified != existing_dir.modified_date or old_created != existing_dir.created_date:
                        updated += 1

                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)

                    try:
                        allowed_file, _ = self.policy.is_path_allowed(file_path)
                        if not allowed_file:
                            continue

                        stat = os.stat(file_path)
                        file_abs = os.path.abspath(file_path)
                        file_key = file_abs.lower()
                        seen_paths.add(file_key)

                        direct_file_bytes += stat.st_size

                        existing_file = self._path_to_metadata.get(file_key)
                        if existing_file is None:
                            metadata = FileMetadata(
                                filename=filename,
                                full_path=file_abs,
                                parent_folder=os.path.dirname(file_abs),
                                extension=Path(filename).suffix.lower(),
                                file_type=self.policy.get_file_type(file_path),
                                size_bytes=stat.st_size,
                                modified_date=datetime.fromtimestamp(stat.st_mtime),
                                created_date=datetime.fromtimestamp(stat.st_ctime),
                            )
                            self.index.append(metadata)
                            self._path_to_metadata[file_key] = metadata
                            added += 1
                        else:
                            new_modified = datetime.fromtimestamp(stat.st_mtime)
                            changed = (
                                existing_file.size_bytes != stat.st_size or
                                existing_file.modified_date != new_modified
                            )
                            existing_file.filename = filename
                            existing_file.parent_folder = os.path.dirname(file_abs)
                            existing_file.extension = Path(filename).suffix.lower()
                            existing_file.file_type = self.policy.get_file_type(file_path)
                            existing_file.size_bytes = stat.st_size
                            existing_file.modified_date = new_modified
                            existing_file.created_date = datetime.fromtimestamp(stat.st_ctime)
                            if changed:
                                updated += 1

                        scanned += 1
                        if progress_callback:
                            progress_callback(scanned, file_path)

                    except (OSError, PermissionError):
                        continue

                existing_dir.size_bytes = direct_file_bytes

            except (OSError, PermissionError):
                continue

        to_remove = existing_under_root - seen_paths
        if to_remove:
            for path_key in to_remove:
                self._path_to_metadata.pop(path_key, None)
            self.index = [m for m in self.index if m.full_path.lower() not in to_remove]

        return {
            "scanned": scanned,
            "added": added,
            "updated": updated,
            "removed": len(to_remove),
        }
    
    def search(
        self,
        query: str,
        extension_filter: Optional[list[str]] = None,
        file_type_filter: Optional[list[str]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        folder_contains: Optional[str] = None,
        max_results: int = 50,
    ) -> list[tuple[FileMetadata, float]]:
        """
        Search the index using lexical/fuzzy matching.
        
        Args:
            query: Search query string
            extension_filter: Filter by file extensions (e.g., ['.pdf', '.md'])
            date_from: Filter files modified after this date
            date_to: Filter files modified before this date
            folder_contains: Filter by folder name containing this string
            max_results: Maximum number of results to return
        
        Returns:
            List of (FileMetadata, score) tuples sorted by score descending
        """
        results: list[tuple[FileMetadata, float]] = []
        query_lower = query.lower()
        query_tokens = query_lower.split() if query_lower else []
        
        for metadata in self.index:
            # Apply filters
            if extension_filter and metadata.extension not in extension_filter:
                continue

            if file_type_filter and metadata.file_type not in file_type_filter:
                continue
            
            if date_from and metadata.modified_date < date_from:
                continue
            
            if date_to and metadata.modified_date > date_to:
                continue
            
            if folder_contains and folder_contains.lower() not in metadata.parent_folder.lower():
                continue
            
            # Calculate score
            score = 0.0
            
            # If no query tokens, return all matching files with base score
            if not query_tokens:
                score = 75.0  # Base score for filtered results
            else:
                # Exact filename match
                if query_lower == metadata.filename.lower():
                    score += 100.0
                
                # Filename starts with query
                elif metadata.filename.lower().startswith(query_lower):
                    score += 50.0
                
                # Filename contains query tokens
                filename_lower = metadata.filename.lower()
                for token in query_tokens:
                    if token in filename_lower:
                        score += 20.0
                    
                    # Fuzzy match (simple substring)
                    if len(token) > 3:
                        # Check for fuzzy containment
                        if self._fuzzy_match(token, filename_lower):
                            score += 10.0
                
                # Path matching
                path_lower = metadata.full_path.lower()
                for token in query_tokens:
                    if token in path_lower:
                        score += 15.0
                
                # Parent folder boost
                if query_lower in metadata.parent_folder.lower():
                    score += 25.0
                
                # Extension match boost if query looks like extension
                if query.startswith('.') and query_lower == metadata.extension:
                    score += 30.0
            
            if score > 0:
                results.append((metadata, score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]
    
    def _fuzzy_match(self, pattern: str, text: str) -> bool:
        """Simple fuzzy matching - checks if pattern chars appear in order in text."""
        pattern_idx = 0
        for char in text:
            if pattern_idx < len(pattern) and char == pattern[pattern_idx]:
                pattern_idx += 1
        return pattern_idx == len(pattern)
    
    def get_by_path(self, file_path: str) -> Optional[FileMetadata]:
        """Get metadata for a specific file path."""
        return self._path_to_metadata.get(file_path.lower())
    
    def get_all_extensions(self) -> set[str]:
        """Get all unique file extensions in the index."""
        return {m.extension for m in self.index}
    
    def get_all_parent_folders(self) -> set[str]:
        """Get all unique parent folders in the index."""
        return {m.parent_folder for m in self.index}
    
    def count(self) -> int:
        """Return total number of indexed files."""
        return len(self.index)

    def save_cache(self, cache_path: str, scanned_directories: Optional[list[str]] = None) -> None:
        """Persist the in-memory metadata index to disk."""
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": 1,
            "saved_at": datetime.utcnow().isoformat(),
            "scanned_directories": scanned_directories or [],
            "items": [m.to_dict() for m in self.index],
        }

        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_cache(self, cache_path: str) -> tuple[int, list[str]]:
        """Load metadata index from disk and replace current in-memory index."""
        path = Path(cache_path)
        if not path.exists():
            return 0, []

        with open(path, "rb") as f:
            payload = pickle.load(f)

        items = payload.get("items", [])
        scanned_directories = payload.get("scanned_directories", [])

        self.index = []
        self._path_to_metadata = {}

        for item in items:
            metadata = FileMetadata(
                filename=item.get("filename", ""),
                full_path=item.get("full_path", ""),
                parent_folder=item.get("parent_folder", ""),
                extension=item.get("extension", ""),
                file_type=item.get("file_type", "other"),
                size_bytes=item.get("size_bytes", 0),
                modified_date=datetime.fromisoformat(item["modified_date"]) if item.get("modified_date") else datetime.utcnow(),
                created_date=datetime.fromisoformat(item["created_date"]) if item.get("created_date") else None,
            )
            self.index.append(metadata)
            self._path_to_metadata[metadata.full_path.lower()] = metadata

        return len(self.index), scanned_directories
