"""
Two-stage retrieval service.

This module implements the core retrieval pipeline:
1. Stage 1: File-level retrieval using metadata search
2. Stage 2: Content-level retrieval for shortlisted files
"""

from typing import Optional
from dataclasses import dataclass, field
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import FileMetadata, SearchResult, Chunk
from core.policy import SafetyPolicy
from indexer.metadata_indexer import MetadataIndexer
from indexer.content_extractor import ContentExtractor, ChunkingConfig


@dataclass
class RetrievalConfig:
    """Configuration for the retrieval service."""
    
    # Stage 1: Metadata search
    max_metadata_results: int = 50
    
    # Stage 2: Content grounding
    max_files_to_read: int = 10
    max_chunks_per_file: int = 5
    min_content_score_threshold: float = 0.5
    
    # Whether to attempt content grounding
    enable_content_grounding: bool = True


class RetrievalService:
    """
    Two-stage retrieval service.
    
    Stage 1 searches file metadata (filename, path, date, extension).
    Stage 2 extracts and searches content from shortlisted files.
    """
    
    def __init__(
        self,
        policy: Optional[SafetyPolicy] = None,
        config: Optional[RetrievalConfig] = None,
    ):
        self.policy = policy or SafetyPolicy()
        self.config = config or RetrievalConfig()
        
        self.indexer = MetadataIndexer(self.policy)
        self.extractor = ContentExtractor(ChunkingConfig())
        
        # Track how many files we've read this session
        self._files_read_count = 0
        self._chars_extracted = 0
    
    def index_directories(self, directories: list[str], progress_callback=None) -> int:
        """
        Index multiple directories.
        
        Args:
            directories: List of root directories to scan
            progress_callback: Optional callback(total_count, current_path)
        
        Returns:
            Total number of files indexed
        """
        total = 0
        for directory in directories:
            callback = None
            if progress_callback is not None:
                def callback(current_count, current_path, running_total=total):
                    progress_callback(running_total + current_count, current_path)

            count = self.indexer.scan_directory(directory, callback)
            total += count
        return total
    
    def retrieve(
        self,
        query: str,
        extension_filter: Optional[list[str]] = None,
        date_from=None,
        date_to=None,
        folder_contains: Optional[str] = None,
        needs_content: bool = False,
        max_results: int = 50,
    ) -> list[SearchResult]:
        """
        Perform two-stage retrieval.
        
        Args:
            query: User's search query
            extension_filter: Filter by file extensions
            date_from: Filter by modification date
            date_to: Filter by modification date
            folder_contains: Filter by folder name
            needs_content: Whether content grounding is needed
        
        Returns:
            List of SearchResult objects
        """
        # Stage 1: Metadata retrieval
        metadata_results = self.indexer.search(
            query=query,
            extension_filter=extension_filter,
            date_from=date_from,
            date_to=date_to,
            folder_contains=folder_contains,
            max_results=self.config.max_metadata_results,
        )
        
        if not metadata_results:
            return []
        
        results: list[SearchResult] = []
        
        # Determine which files to read for content grounding
        files_to_read = []
        for metadata, score in metadata_results:
            if len(files_to_read) >= self.config.max_files_to_read:
                break
            
            # Check if we can read this file's content
            if needs_content and self.config.enable_content_grounding:
                can_read, _ = self.policy.can_read_content(
                    metadata.full_path, 
                    metadata.size_bytes
                )
                if can_read:
                    files_to_read.append((metadata, score))
            else:
                # Metadata-only result
                results.append(SearchResult(
                    metadata=metadata,
                    score=score,
                    match_type="metadata_only",
                ))
        
        # Stage 2: Content grounding for shortlisted files
        for metadata, base_score in files_to_read:
            content_results = self._ground_in_content(query, metadata, base_score)
            if content_results:
                results.extend(content_results)
            else:
                # Fall back to metadata-only if content extraction failed
                results.append(SearchResult(
                    metadata=metadata,
                    score=base_score * 0.8,  # Slight penalty for no content
                    match_type="metadata_only",
                ))
        
        # Sort all results by score
        results.sort(key=lambda r: r.score, reverse=True)
        
        return results
    
    def _ground_in_content(
        self, 
        query: str, 
        metadata: FileMetadata,
        base_score: float,
    ) -> list[SearchResult]:
        """
        Extract content from a file and find relevant chunks.
        
        Args:
            query: Original query
            metadata: File metadata
            base_score: Score from metadata matching
        
        Returns:
            List of SearchResult objects with content snippets
        """
        # Check limits
        if self._files_read_count >= self.policy.max_files_per_query:
            return []
        
        # Extract text
        text = self.extractor.extract_text(metadata.full_path)
        if not text:
            return []
        
        self._files_read_count += 1
        self._chars_extracted += len(text)
        
        if self._chars_extracted > self.policy.max_extracted_chars:
            return []
        
        # Determine if this is code
        is_code = metadata.file_type == "code"
        
        # Chunk the content
        chunks = self.extractor.chunk_text(text, metadata.full_path, is_code=is_code)
        
        if not chunks:
            return []
        
        # Score chunks against query (simple lexical scoring)
        query_lower = query.lower()
        query_tokens = query_lower.split()
        
        scored_chunks: list[tuple[Chunk, float]] = []
        
        for chunk in chunks:
            chunk_text_lower = chunk.text.lower()
            score = 0.0
            
            # Exact phrase match
            if query_lower in chunk_text_lower:
                score += 50.0
            
            # Token matches
            for token in query_tokens:
                if token in chunk_text_lower:
                    score += 10.0
            
            # Boost if chunk contains query tokens near the beginning
            preview = chunk_text_lower[:200]
            for token in query_tokens:
                if token in preview:
                    score += 5.0
            
            if score > 0:
                scored_chunks.append((chunk, score))
        
        # Sort by score and take top chunks
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        top_chunks = scored_chunks[:self.config.max_chunks_per_file]
        
        if not top_chunks:
            return []
        
        # Create search results with snippets
        results = []
        for chunk, chunk_score in top_chunks:
            # Combine base metadata score with chunk score
            combined_score = (base_score * 0.4) + (chunk_score * 0.6)
            
            # Extract snippet (first 300 chars of chunk)
            snippet = chunk.text[:300]
            if len(chunk.text) > 300:
                snippet += "..."
            
            results.append(SearchResult(
                metadata=metadata,
                score=combined_score,
                match_type="content_grounding",
                snippets=[snippet],
            ))
        
        return results
    
    def reset_limits(self):
        """Reset file reading limits for a new query."""
        self._files_read_count = 0
        self._chars_extracted = 0
    
    def get_index_stats(self) -> dict:
        """Get statistics about the index."""
        extensions = self.indexer.get_all_extensions()
        return {
            "total_files": self.indexer.count(),
            "unique_extensions": len(extensions),
            "extensions": sorted(list(extensions)),
        }
