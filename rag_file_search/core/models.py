"""
Metadata-first RAG File Search System

This system implements a two-stage retrieval pipeline:
1. File-level retrieval using metadata (filename, path, date, extension)
2. Content-level retrieval only for shortlisted, policy-approved files
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class FileMetadata:
    """Represents indexed metadata for a file."""
    
    filename: str
    full_path: str
    parent_folder: str
    extension: str
    file_type: str  # text, image, video, code, document, other
    size_bytes: int
    modified_date: datetime
    created_date: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "full_path": self.full_path,
            "parent_folder": self.parent_folder,
            "extension": self.extension,
            "file_type": self.file_type,
            "size_bytes": self.size_bytes,
            "modified_date": self.modified_date.isoformat() if self.modified_date else None,
            "created_date": self.created_date.isoformat() if self.created_date else None,
        }


@dataclass
class SearchResult:
    """Result from metadata or content retrieval."""
    
    metadata: FileMetadata
    score: float
    match_type: str  # metadata_only, content_grounding
    snippets: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata.to_dict(),
            "score": self.score,
            "match_type": self.match_type,
            "snippets": self.snippets,
        }


@dataclass
class Chunk:
    """A chunk of text extracted from a file for grounding."""
    
    file_path: str
    chunk_id: str
    text: str
    start_offset: int
    end_offset: int
    metadata: dict = field(default_factory=dict)
