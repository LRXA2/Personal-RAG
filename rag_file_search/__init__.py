"""
RAG File Search System

A metadata-first, two-stage retrieval system for searching files safely.

Usage:
    from rag_file_search import RagFileSearch
    
    # Initialize with your directories
    searcher = RagFileSearch(allowed_roots=["D:/", "E:/"])
    
    # Index your files
    searcher.index()
    
    # Search
    results = searcher.search("Find my project documentation")
    print(results)
"""

from .core.policy import SafetyPolicy
from .core.retrieval_service import RetrievalService, RetrievalConfig
from .core.planner import LLMPlanner
from .core.models import FileMetadata, SearchResult, Chunk


class RagFileSearch:
    """
    Main entry point for the RAG file search system.
    
    Provides a simple interface for:
    - Indexing directories
    - Searching files with natural language queries
    - Retrieving content with safety controls
    """
    
    def __init__(
        self,
        allowed_roots: list[str] = None,
        max_files_to_read: int = 10,
        enable_content_grounding: bool = True,
        semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        semantic_trust_remote_code: bool = False,
        semantic_query_prompt_name: str = None,
        semantic_document_prompt_name: str = None,
        reranker_model_name: str = None,
        enable_reranker: bool = False,
        reranker_trust_remote_code: bool = False,
        reranker_weight: float = 0.25,
        enable_llm_query_planner: bool = False,
        query_planner_model_name: str = "Qwen/Qwen3.5-4B",
        query_planner_trust_remote_code: bool = True,
        enable_personal_intent_prefilter: bool = True,
        enable_expansion_queries: bool = False,
        disable_top_folder_expansion_for_generic_personal: bool = True,
        personal_profile_default: str = "balanced",
    ):
        """
        Initialize the file search system.
        
        Args:
            allowed_roots: Root directories to scan (e.g., ["D:/", "E:/"])
            max_files_to_read: Max files to open per query
            enable_content_grounding: Whether to read file contents for grounding
            semantic_model_name: SentenceTransformer model for metadata embeddings
            reranker_model_name: Optional CrossEncoder reranker model
            enable_llm_query_planner: Whether to use a local LLM for query planning
            query_planner_model_name: Local/Hugging Face model id for query planning
            enable_personal_intent_prefilter: Deterministic personal-doc prefilter
            enable_expansion_queries: Allow LLM expansion queries in ranking
            personal_profile_default: strict | balanced | off
        """
        # Configure safety policy
        policy = SafetyPolicy(
            allowed_roots=allowed_roots or [],
        )
        
        # Configure retrieval
        config = RetrievalConfig(
            max_metadata_results=50,
            max_files_to_read=max_files_to_read,
            max_chunks_per_file=5,
            enable_content_grounding=enable_content_grounding,
            semantic_model_name=semantic_model_name,
            semantic_trust_remote_code=semantic_trust_remote_code,
            semantic_query_prompt_name=semantic_query_prompt_name,
            semantic_document_prompt_name=semantic_document_prompt_name,
            enable_cross_encoder_reranking=enable_reranker,
            reranker_model_name=reranker_model_name,
            reranker_trust_remote_code=reranker_trust_remote_code,
            reranker_weight=reranker_weight,
            enable_llm_query_planner=enable_llm_query_planner,
            query_planner_model_name=query_planner_model_name,
            query_planner_trust_remote_code=query_planner_trust_remote_code,
            enable_personal_intent_prefilter=enable_personal_intent_prefilter,
            enable_expansion_queries=enable_expansion_queries,
            disable_top_folder_expansion_for_generic_personal=disable_top_folder_expansion_for_generic_personal,
            personal_profile_default=personal_profile_default,
        )
        
        # Initialize services
        self.retrieval_service = RetrievalService(policy=policy, config=config)
        self.planner = LLMPlanner()
        
        self._indexed = False
    
    def index(self, directories: list[str] = None, verbose: bool = True) -> int:
        """
        Index files in the specified directories.
        
        Args:
            directories: Directories to scan. If None, uses allowed_roots.
            verbose: Print progress information
        
        Returns:
            Number of files indexed
        """
        if directories is None:
            directories = self.retrieval_service.policy.allowed_roots
        
        if not directories:
            raise ValueError("No directories specified. Provide directories or set allowed_roots.")
        
        if verbose:
            print(f"Indexing {len(directories)} director(ies)...")
            
            def progress_callback(count, path):
                if count % 1000 == 0:
                    print(f"  Indexed {count} files... (current: {path})")
            
            total = self.retrieval_service.index_directories(
                directories, 
                progress_callback=progress_callback
            )
            print(f"Indexing complete. Total files: {total}")
        else:
            total = self.retrieval_service.index_directories(directories)
        
        self._indexed = True
        return total
    
    def search(
        self,
        query: str,
        max_results: int = 20,
        metadata_only: bool = False,
    ) -> list[SearchResult]:
        """
        Search for files matching the query.
        
        Args:
            query: Natural language search query
            max_results: Maximum number of results to return
            metadata_only: If True, skip content grounding
        
        Returns:
            List of SearchResult objects
        """
        if not self._indexed:
            raise RuntimeError("Must call index() before searching")
        
        # Parse query intent
        intent = self.planner.parse_query(query)
        
        # Build keyword query
        keyword_query = " ".join(intent.keywords) if intent.keywords else query
        
        # Perform retrieval
        self.retrieval_service.reset_limits()
        results = self.retrieval_service.retrieve(
            query=keyword_query,
            extension_filter=intent.extensions,
            date_from=intent.date_from,
            date_to=intent.date_to,
            folder_contains=intent.folder_contains,
            needs_content=not metadata_only and intent.needs_content,
        )
        
        return results[:max_results]
    
    def answer(self, query: str) -> str:
        """
        Get a formatted answer to a natural language query.
        
        Args:
            query: Natural language question or search request
        
        Returns:
            Formatted answer string with file information
        """
        if not self._indexed:
            raise RuntimeError("Must call index() before searching")
        
        # Parse query intent
        intent = self.planner.parse_query(query)
        
        # Build keyword query
        keyword_query = " ".join(intent.keywords) if intent.keywords else query
        
        # Perform retrieval
        self.retrieval_service.reset_limits()
        results = self.retrieval_service.retrieve(
            query=keyword_query,
            extension_filter=intent.extensions,
            date_from=intent.date_from,
            date_to=intent.date_to,
            folder_contains=intent.folder_contains,
            needs_content=intent.needs_content,
        )
        
        # Format answer
        return self.planner.format_answer(query, intent, results)
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        return self.retrieval_service.get_index_stats()
    
    def add_policy_exception(
        self,
        path_pattern: str,
        allow_content_reading: bool = True,
    ) -> None:
        """
        Add an exception to the safety policy.
        
        Use with caution - this allows content reading for specific paths.
        
        Args:
            path_pattern: Path or pattern to allow
            allow_content_reading: Whether to allow content reading
        """
        # This could be extended to support custom policy rules
        # For now, it's a placeholder for future extensibility
        pass


__all__ = [
    "RagFileSearch",
    "SafetyPolicy",
    "RetrievalService",
    "RetrievalConfig",
    "LLMPlanner",
    "FileMetadata",
    "SearchResult",
    "Chunk",
]
