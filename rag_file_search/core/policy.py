"""
Policy layer for controlling file access and safety.

This module enforces:
- Path sanitization and traversal protection
- Directory allowlists/blocklists
- File extension filters
- Size limits
- Filename pattern matching for sensitive content
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SafetyPolicy:
    """Defines safety constraints for file access."""
    
    # Root directories to scan (e.g., D:/, E:/)
    allowed_roots: list[str] = field(default_factory=list)
    
    # Directories to explicitly block (case-insensitive)
    blocked_dirs: list[str] = field(default_factory=lambda: [
        "windows", "program files", "appdata", "$recycle.bin",
        "system volume information", "config.msi",
        ".git", "node_modules", "__pycache__",
        "secrets", "credentials", ".ssh", ".gnupg",
    ])
    
    # Filename patterns that suggest sensitive content (regex)
    sensitive_filename_patterns: list[str] = field(default_factory=lambda: [
        r".*password.*",
        r".*secret.*",
        r".*credential.*",
        r".*private.*",
        r".*key$",  # Files ending in 'key'
        r"^\.env.*",
        r".*\.pem$",
        r".*\.key$",
    ])
    
    # Extensions allowed for content reading
    content_reading_extensions: list[str] = field(default_factory=lambda: [
        ".txt", ".md", ".rst",
        ".pdf", ".docx", ".doc",
        ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".go", ".rs",
        ".json", ".yaml", ".yml", ".toml", ".xml",
        ".csv", ".log",
    ])
    
    # Extensions blocked entirely
    blocked_extensions: list[str] = field(default_factory=lambda: [
        ".exe", ".dll", ".so", ".bin", ".bat", ".sh",
        ".msi", ".cmd", ".ps1",
    ])
    
    # Maximum file size for content reading (default: 10 MB)
    max_file_size_bytes: int = 10 * 1024 * 1024
    
    # Maximum number of files to open per query
    max_files_per_query: int = 10
    
    # Maximum total extracted text (tokens approximated by chars / 4)
    max_extracted_chars: int = 50000
    
    def is_path_allowed(self, file_path: str) -> tuple[bool, Optional[str]]:
        """
        Check if a file path is allowed under the safety policy.
        
        Returns:
            (allowed, reason): Tuple of whether access is allowed and why not if blocked.
        """
        path = Path(file_path).resolve()
        path_str = str(path).lower()
        
        # Check if path is under any allowed root
        if self.allowed_roots:
            is_under_root = any(
                path_str.startswith(str(Path(root).resolve()).lower())
                for root in self.allowed_roots
            )
            if not is_under_root:
                return False, f"Path not under allowed roots: {self.allowed_roots}"
        
        # Check for path traversal attempts
        if ".." in str(path):
            return False, "Path traversal detected"
        
        # Check each component of the path against blocked dirs
        for part in path.parts:
            if part.lower() in self.blocked_dirs:
                return False, f"Path contains blocked directory: {part}"
        
        # Check extension
        ext = path.suffix.lower()
        if ext in self.blocked_extensions:
            return False, f"Blocked extension: {ext}"
        
        # Check filename patterns
        filename = path.name.lower()
        for pattern in self.sensitive_filename_patterns:
            if re.match(pattern, filename, re.IGNORECASE):
                return False, f"Filename matches sensitive pattern: {pattern}"
        
        return True, None
    
    def can_read_content(self, file_path: str, file_size: int) -> tuple[bool, Optional[str]]:
        """
        Check if content can be read from a file.
        
        Returns:
            (allowed, reason): Tuple of whether reading is allowed and why not if blocked.
        """
        # First check general path allowance
        allowed, reason = self.is_path_allowed(file_path)
        if not allowed:
            return False, reason
        
        path = Path(file_path)
        ext = path.suffix.lower()
        
        # Check if extension allows content reading
        if ext not in self.content_reading_extensions:
            return False, f"Extension not allowed for content reading: {ext}"
        
        # Check file size
        if file_size > self.max_file_size_bytes:
            return False, f"File too large: {file_size} > {self.max_file_size_bytes}"
        
        return True, None
    
    def get_file_type(self, file_path: str) -> str:
        """Categorize file type based on extension."""
        ext = Path(file_path).suffix.lower()
        
        text_exts = {".txt", ".md", ".rst", ".log"}
        code_exts = {".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".go", ".rs", ".rb", ".php"}
        doc_exts = {".pdf", ".docx", ".doc", ".odt", ".rtf"}
        data_exts = {".json", ".yaml", ".yml", ".toml", ".xml", ".csv"}
        image_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp"}
        video_exts = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"}
        
        if ext in text_exts:
            return "text"
        elif ext in code_exts:
            return "code"
        elif ext in doc_exts:
            return "document"
        elif ext in data_exts:
            return "data"
        elif ext in image_exts:
            return "image"
        elif ext in video_exts:
            return "video"
        else:
            return "other"
