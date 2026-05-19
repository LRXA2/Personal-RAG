"""
FastAPI-based API endpoints for the RAG file search system.

Endpoints:
- POST /search - Search metadata only (simplified)
- GET /download - Download a file
- POST /retrieve/files - Search metadata only (legacy)
- POST /read/file - Read a specific file (with policy checks)
- POST /extract/chunks - Extract chunks from a file
- POST /answer - Full retrieval + answer pipeline
- GET /stats - Index statistics
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import time
import threading
import os
import sys
import tempfile
import zipfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.policy import SafetyPolicy
from core.retrieval_service import RetrievalService, RetrievalConfig
from core.planner import LLMPlanner


DEFAULT_ALLOWED_ROOTS = ["D:/", "E:/"]
DEFAULT_INDEX_CACHE_PATH = ".cache/metadata_index.pkl"
PUBLIC_MODE = os.getenv("RAG_PUBLIC_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}


# Request/Response models

class SearchRequest(BaseModel):
    query: str
    max_results: int = 50
    document_type: str = "all"
    debug: bool = False
    personal_profile: Optional[str] = None
    allowed_file_types: Optional[List[str]] = None
    blocked_extensions: Optional[List[str]] = None


class FileMetadataSimple(BaseModel):
    filename: str
    path: str
    item_type: str = "file"
    extension: str
    size: int
    modified_date: str
    relevance_score: Optional[float] = None
    match_reasons: List[str] = Field(default_factory=list)
    score_breakdown: dict[str, float] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    results: List[FileMetadataSimple]
    search_time_ms: float
    total_found: int
    debug: Optional[dict] = None


class RetrieveFilesRequest(BaseModel):
    query: str
    document_type: str = "all"
    extension_filter: Optional[List[str]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    folder_contains: Optional[str] = None
    max_results: int = 50
    debug: bool = False
    personal_profile: Optional[str] = None
    allowed_file_types: Optional[List[str]] = None
    blocked_extensions: Optional[List[str]] = None


class FileMetadataResponse(BaseModel):
    filename: str
    full_path: str
    parent_folder: str
    extension: str
    file_type: str
    size_bytes: int
    modified_date: str
    created_date: Optional[str] = None


class SearchResultResponse(BaseModel):
    metadata: FileMetadataResponse
    score: float
    match_type: str
    snippets: List[str] = []


class RetrieveFilesResponse(BaseModel):
    results: List[SearchResultResponse]
    total_found: int
    debug: Optional[dict] = None


class ReadFileRequest(BaseModel):
    file_path: str


class ReadFileResponse(BaseModel):
    file_path: str
    content: str
    truncated: bool = False
    char_count: int


class ExtractChunksRequest(BaseModel):
    file_path: str
    is_code: bool = False


class ChunkResponse(BaseModel):
    chunk_id: str
    text: str
    start_offset: int
    end_offset: int


class ExtractChunksResponse(BaseModel):
    file_path: str
    chunks: List[ChunkResponse]
    total_chunks: int


class AnswerRequest(BaseModel):
    query: str


class AnswerResponse(BaseModel):
    query: str
    answer: str
    sources: List[FileMetadataResponse] = []


class StatsResponse(BaseModel):
    total_files: int
    unique_extensions: int
    extensions: List[str]


class ScanDirectoriesRequest(BaseModel):
    directories: List[str] = Field(default_factory=list)
    incremental: bool = True


class IndexStatusResponse(BaseModel):
    is_indexing: bool
    indexed_files: int
    current_path: Optional[str] = None
    directories: List[str] = Field(default_factory=list)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None


# Create FastAPI app

app = FastAPI(
    title="RAG File Search API",
    description="Metadata-first file search with selective content grounding",
    version="1.0.0",
    docs_url=None if PUBLIC_MODE else "/docs",
    redoc_url=None if PUBLIC_MODE else "/redoc",
    openapi_url=None if PUBLIC_MODE else "/openapi.json",
)

# Initialize services (will be configured on startup)
retrieval_service: Optional[RetrievalService] = None
planner: Optional[LLMPlanner] = None


# Global policy variable - can be overridden by run_ui.py
policy: Optional[SafetyPolicy] = None
# Directories to scan on startup (set by run_ui.py)
_dirs_to_scan_on_startup: list[str] = []

_index_status_lock = threading.Lock()
_index_status = {
    "is_indexing": False,
    "indexed_files": 0,
    "current_path": None,
    "directories": [],
    "started_at": None,
    "finished_at": None,
    "error": None,
}


def _utcnow_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _get_allowed_roots() -> list[str]:
    """Get allowed roots from env var or fallback defaults."""
    raw = os.getenv("RAG_ALLOWED_ROOTS", "")
    if raw.strip():
        return [p.strip() for p in raw.split(",") if p.strip()]
    return list(DEFAULT_ALLOWED_ROOTS)


def _get_default_scan_directories() -> list[str]:
    """Get default scan directories from env var or fallback defaults."""
    raw = os.getenv("RAG_DEFAULT_SCAN_DIRS", "")
    if raw.strip():
        return [p.strip() for p in raw.split(",") if p.strip()]
    return _get_allowed_roots()


def _get_index_cache_path() -> str:
    """Get index cache path from env var or default."""
    return os.getenv("RAG_INDEX_CACHE_PATH", DEFAULT_INDEX_CACHE_PATH)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _enforce_public_endpoint(path: str) -> None:
    """Hide non-public endpoints when running in public mode."""
    if not PUBLIC_MODE:
        return
    allowed = {"/search", "/download", "/index/scan", "/index/reset"}
    if path not in allowed:
        raise HTTPException(status_code=404, detail="Not found")


def _validate_scan_directories(directories: List[str], active_policy: SafetyPolicy) -> list[str]:
    """Validate and normalize directories for scan requests."""
    normalized: list[str] = []
    seen: set[str] = set()

    for directory in directories:
        candidate = (directory or "").strip()
        if not candidate:
            continue

        path = Path(candidate).resolve()
        if not path.exists():
            raise HTTPException(status_code=400, detail=f"Directory does not exist: {candidate}")
        if not path.is_dir():
            raise HTTPException(status_code=400, detail=f"Path is not a directory: {candidate}")

        allowed, reason = active_policy.is_path_allowed(str(path))
        if not allowed:
            raise HTTPException(status_code=403, detail=f"Directory not allowed: {candidate} ({reason})")

        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            normalized.append(str(path))

    if not normalized:
        raise HTTPException(status_code=400, detail="No directories provided")

    return normalized


def _normalize_extension(value: str) -> str:
    ext = (value or "").strip().lower()
    if not ext:
        return ""
    return ext if ext.startswith(".") else f".{ext}"


def _resolve_document_filters(
    document_type: str,
    extension_filter: Optional[List[str]] = None,
) -> tuple[Optional[list[str]], Optional[list[str]]]:
    """
    Resolve API filters from document_type and extension_filter.

    `document_type` supports comma-separated tokens such as:
      - broad types: folder, text, code, document, data, image, video, other
      - extensions: docx, txt, pdf (with or without leading dot)
      - all / * / any: no filtering
    """
    file_types: set[str] = set()
    extensions: set[str] = set()

    if extension_filter:
        for ext in extension_filter:
            normalized = _normalize_extension(ext)
            if normalized:
                extensions.add(normalized)

    raw = (document_type or "all").strip().lower()
    if raw in {"all", "*", "any", ""}:
        return None, sorted(list(extensions)) if extensions else None

    tokens = [t.strip().lower() for t in raw.split(",") if t.strip()]
    if not tokens:
        return None, sorted(list(extensions)) if extensions else None

    allowed_file_types = {
        "folder", "text", "code", "document", "data", "image", "video", "other",
    }

    # Common aliases for convenience
    alias_map = {
        "folders": "folder",
        "docs": "document",
        "images": "image",
        "videos": "video",
    }

    for token in tokens:
        token = alias_map.get(token, token)
        if token in allowed_file_types:
            file_types.add(token)
            continue

        normalized_ext = _normalize_extension(token)
        if normalized_ext:
            extensions.add(normalized_ext)
            continue

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid document_type token. Use comma-separated values like: "
                "all, folder, text, code, document, data, image, video, other, "
                "or extensions such as txt, docx, pdf"
            ),
        )

    file_type_filter = sorted(list(file_types)) if file_types else None
    extension_filter_resolved = sorted(list(extensions)) if extensions else None
    return file_type_filter, extension_filter_resolved


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global retrieval_service, planner, policy
    
    # Use existing policy if set, otherwise create default from config/env
    if policy is None:
        policy = SafetyPolicy(
            allowed_roots=_get_allowed_roots(),
        )
    
    config = RetrievalConfig(
        max_metadata_results=50,
        max_files_to_read=10,
        max_chunks_per_file=5,
        semantic_model_name=os.getenv("RAG_SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        semantic_trust_remote_code=_env_bool("RAG_SEMANTIC_TRUST_REMOTE_CODE", False),
        semantic_query_prompt_name=os.getenv("RAG_SEMANTIC_QUERY_PROMPT") or None,
        semantic_document_prompt_name=os.getenv("RAG_SEMANTIC_DOCUMENT_PROMPT") or None,
        lexical_weight=0.6,
        semantic_weight=0.4,
        enable_cross_encoder_reranking=_env_bool("RAG_ENABLE_RERANKER", False),
        reranker_model_name=os.getenv("RAG_RERANKER_MODEL") or None,
        reranker_trust_remote_code=_env_bool("RAG_RERANKER_TRUST_REMOTE_CODE", False),
        reranker_top_k=_env_int("RAG_RERANKER_TOP_K", 50),
        reranker_weight=_env_float("RAG_RERANKER_WEIGHT", 0.25),
        enable_llm_query_planner=_env_bool("RAG_ENABLE_LLM_QUERY_PLANNER", False),
        query_planner_model_name=os.getenv("RAG_QUERY_PLANNER_MODEL", "Qwen/Qwen3.5-4B"),
        query_planner_trust_remote_code=_env_bool("RAG_QUERY_PLANNER_TRUST_REMOTE_CODE", True),
        query_planner_max_new_tokens=_env_int("RAG_QUERY_PLANNER_MAX_NEW_TOKENS", 512),
        query_planner_temperature=_env_float("RAG_QUERY_PLANNER_TEMPERATURE", 0.0),
        enable_personal_intent_prefilter=_env_bool("RAG_ENABLE_PERSONAL_INTENT_PREFILTER", True),
        enable_expansion_queries=_env_bool("RAG_ENABLE_EXPANSION_QUERIES", False),
        disable_top_folder_expansion_for_generic_personal=_env_bool(
            "RAG_DISABLE_TOP_FOLDER_EXPANSION_FOR_GENERIC_PERSONAL",
            True,
        ),
        personal_profile_default=(os.getenv("RAG_PERSONAL_PROFILE_DEFAULT", "balanced") or "balanced").strip().lower(),
    )
    
    retrieval_service = RetrievalService(policy=policy, config=config)
    planner = LLMPlanner()

    # Load cached index if available
    cache_path = _get_index_cache_path()
    try:
        loaded_count, cached_dirs = retrieval_service.load_index_cache(cache_path)
        if loaded_count > 0:
            with _index_status_lock:
                _index_status["is_indexing"] = False
                _index_status["indexed_files"] = loaded_count
                _index_status["current_path"] = None
                _index_status["directories"] = cached_dirs
                _index_status["started_at"] = None
                _index_status["finished_at"] = _utcnow_iso()
                _index_status["error"] = None
            print(f"[OK] Loaded cached index with {loaded_count} items", flush=True)
    except Exception as exc:
        print(f"[WARN] Failed to load index cache: {exc}", flush=True)
    
    # Auto-scan directories in the background if specified
    if _dirs_to_scan_on_startup and retrieval_service:
        dirs = list(_dirs_to_scan_on_startup)

        with _index_status_lock:
            _index_status["is_indexing"] = True
            _index_status["indexed_files"] = 0
            _index_status["current_path"] = None
            _index_status["directories"] = dirs
            _index_status["started_at"] = _utcnow_iso()
            _index_status["finished_at"] = None
            _index_status["error"] = None

        print(f"\n[INFO] Scanning directories in background: {', '.join(dirs)}", flush=True)
        print("  [INFO] API is available while indexing runs.", flush=True)

        def _index_worker():
            try:
                def _progress(total_count: int, current_path: str):
                    with _index_status_lock:
                        _index_status["indexed_files"] = total_count
                        _index_status["current_path"] = current_path
                    if total_count % 500 == 0:
                        print(f"[INFO] Indexed {total_count} files...", flush=True)

                count = retrieval_service.index_directories(dirs, progress_callback=_progress)
                retrieval_service.save_index_cache(_get_index_cache_path(), scanned_directories=dirs)

                with _index_status_lock:
                    _index_status["is_indexing"] = False
                    _index_status["indexed_files"] = count
                    _index_status["finished_at"] = _utcnow_iso()

                print(f"[OK] Indexed {count} files", flush=True)
                print("[TIP] Use the UI search bar or call POST /index/scan to add more directories", flush=True)
            except Exception as exc:
                with _index_status_lock:
                    _index_status["is_indexing"] = False
                    _index_status["error"] = str(exc)
                    _index_status["finished_at"] = _utcnow_iso()
                print(f"[ERROR] Indexing failed: {exc}", flush=True)

        worker = threading.Thread(target=_index_worker, daemon=True)
        worker.start()


@app.post("/retrieve/files", response_model=RetrieveFilesResponse)
async def retrieve_files(request: RetrieveFilesRequest):
    """
    Search file metadata only (no content reading).
    
    Returns candidate files matching the query based on filename, path, 
    extension, and date filters.
    """
    if retrieval_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    _enforce_public_endpoint("/retrieve/files")

    file_type_filter, extension_filter = _resolve_document_filters(
        request.document_type,
        request.extension_filter,
    )

    retrieval_service.reset_limits()
    
    results = retrieval_service.retrieve(
        query=request.query,
        extension_filter=extension_filter,
        file_type_filter=file_type_filter,
        date_from=request.date_from,
        date_to=request.date_to,
        folder_contains=request.folder_contains,
        needs_content=False,  # Metadata only
        personal_profile=request.personal_profile,
        allowed_file_types_override=request.allowed_file_types,
        blocked_extensions_override=request.blocked_extensions,
    )
    
    debug_payload = retrieval_service.get_last_debug_info() if request.debug else None

    return RetrieveFilesResponse(
        results=[
            SearchResultResponse(
                metadata=FileMetadataResponse(
                    filename=r.metadata.filename,
                    full_path=r.metadata.full_path,
                    parent_folder=r.metadata.parent_folder,
                    extension=r.metadata.extension,
                    file_type=r.metadata.file_type,
                    size_bytes=r.metadata.size_bytes,
                    modified_date=r.metadata.modified_date.isoformat(),
                    created_date=r.metadata.created_date.isoformat() if r.metadata.created_date else None,
                ),
                score=r.score,
                match_type=r.match_type,
                snippets=r.snippets,
            )
            for r in results
        ],
        total_found=len(results),
        debug=debug_payload,
    )


@app.post("/read/file", response_model=ReadFileResponse)
async def read_file(request: ReadFileRequest):
    """
    Read content from a specific file (subject to policy checks).
    
    The file must pass all safety policy checks before content is returned.
    """
    if retrieval_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    _enforce_public_endpoint("/read/file")
    
    # Check policy
    can_read, reason = retrieval_service.policy.can_read_content(
        request.file_path,
        0,  # Size check will happen during actual read
    )
    
    if not can_read:
        raise HTTPException(status_code=403, detail=f"Access denied: {reason}")
    
    # Try to read the file
    from pathlib import Path
    try:
        path = Path(request.file_path)
        
        # Get actual size
        file_size = path.stat().st_size
        if file_size > retrieval_service.policy.max_file_size_bytes:
            raise HTTPException(
                status_code=403,
                detail=f"File too large: {file_size} bytes"
            )
        
        # Read content
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        truncated = False
        if len(content) > retrieval_service.policy.max_extracted_chars:
            content = content[:retrieval_service.policy.max_extracted_chars]
            truncated = True
        
        return ReadFileResponse(
            file_path=request.file_path,
            content=content,
            truncated=truncated,
            char_count=len(content),
        )
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Unable to decode file as text")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")


@app.post("/extract/chunks", response_model=ExtractChunksResponse)
async def extract_chunks(request: ExtractChunksRequest):
    """
    Extract text chunks from a file for grounding.
    
    Chunks are created following the configured chunking strategy.
    """
    if retrieval_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    _enforce_public_endpoint("/extract/chunks")
    
    # First check if we can read this file
    can_read, reason = retrieval_service.policy.can_read_content(
        request.file_path,
        0,
    )
    
    if not can_read:
        raise HTTPException(status_code=403, detail=f"Access denied: {reason}")
    
    # Extract and chunk
    text = retrieval_service.extractor.extract_text(request.file_path)
    if text is None:
        raise HTTPException(status_code=400, detail="Unable to extract text from file")
    
    chunks = retrieval_service.extractor.chunk_text(
        text=text,
        file_path=request.file_path,
        is_code=request.is_code,
    )
    
    return ExtractChunksResponse(
        file_path=request.file_path,
        chunks=[
            ChunkResponse(
                chunk_id=c.chunk_id,
                text=c.text,
                start_offset=c.start_offset,
                end_offset=c.end_offset,
            )
            for c in chunks
        ],
        total_chunks=len(chunks),
    )


@app.post("/answer", response_model=AnswerResponse)
async def answer(request: AnswerRequest):
    """
    Full pipeline: parse query -> retrieve files -> ground in content -> format answer.
    
    This is the main endpoint for natural language queries.
    """
    if retrieval_service is None or planner is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    _enforce_public_endpoint("/answer")
    
    # Parse query intent
    intent = planner.parse_query(request.query)
    
    # Build keyword query from intent
    keyword_query = " ".join(intent.keywords) if intent.keywords else request.query
    
    # Perform retrieval
    retrieval_service.reset_limits()
    results = retrieval_service.retrieve(
        query=keyword_query,
        extension_filter=intent.extensions,
        date_from=intent.date_from,
        date_to=intent.date_to,
        folder_contains=intent.folder_contains,
        needs_content=intent.needs_content,
    )
    
    # Format answer
    answer_text = planner.format_answer(request.query, intent, results)
    
    # Collect source metadata
    sources = []
    seen_paths = set()
    for r in results[:10]:
        if r.metadata.full_path not in seen_paths:
            seen_paths.add(r.metadata.full_path)
            sources.append(FileMetadataResponse(
                filename=r.metadata.filename,
                full_path=r.metadata.full_path,
                parent_folder=r.metadata.parent_folder,
                extension=r.metadata.extension,
                file_type=r.metadata.file_type,
                size_bytes=r.metadata.size_bytes,
                modified_date=r.metadata.modified_date.isoformat(),
                created_date=r.metadata.created_date.isoformat() if r.metadata.created_date else None,
            ))
    
    return AnswerResponse(
        query=request.query,
        answer=answer_text,
        sources=sources,
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get index statistics."""
    _enforce_public_endpoint("/stats")
    if retrieval_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    stats = retrieval_service.get_index_stats()
    return StatsResponse(
        total_files=stats["total_files"],
        unique_extensions=stats["unique_extensions"],
        extensions=stats["extensions"],
    )


@app.post("/index/scan")
async def scan_directories(request: ScanDirectoriesRequest):
    """
    Scan and index directories.
    
    Returns the total number of files indexed.
    """
    if retrieval_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    requested_dirs = [d for d in request.directories if d and d.strip()]
    if requested_dirs:
        directories = _validate_scan_directories(requested_dirs, retrieval_service.policy)
    else:
        directories = _validate_scan_directories(_get_default_scan_directories(), retrieval_service.policy)

    incremental = request.incremental

    with _index_status_lock:
        if _index_status["is_indexing"]:
            raise HTTPException(status_code=409, detail="Indexing already in progress")

        _index_status["is_indexing"] = True
        _index_status["indexed_files"] = 0
        _index_status["current_path"] = None
        _index_status["directories"] = directories
        _index_status["started_at"] = _utcnow_iso()
        _index_status["finished_at"] = None
        _index_status["error"] = None

    def _index_worker():
        try:
            def _progress(total_count: int, current_path: str):
                with _index_status_lock:
                    _index_status["indexed_files"] = total_count
                    _index_status["current_path"] = current_path
                if total_count % 500 == 0:
                    print(f"[INFO] Indexed {total_count} files...", flush=True)

            if incremental:
                refresh_stats = retrieval_service.refresh_directories(directories, progress_callback=_progress)
                count = retrieval_service.get_index_stats()["total_files"]
                print(
                    f"[OK] Incremental refresh done: +{refresh_stats['added']} ~{refresh_stats['updated']} -{refresh_stats['removed']}",
                    flush=True,
                )
            else:
                count = retrieval_service.index_directories(directories, progress_callback=_progress)
            retrieval_service.save_index_cache(_get_index_cache_path(), scanned_directories=directories)

            with _index_status_lock:
                _index_status["is_indexing"] = False
                _index_status["indexed_files"] = count
                _index_status["finished_at"] = _utcnow_iso()

            print(f"[OK] Indexed {count} files", flush=True)
        except Exception as exc:
            with _index_status_lock:
                _index_status["is_indexing"] = False
                _index_status["error"] = str(exc)
                _index_status["finished_at"] = _utcnow_iso()
            print(f"[ERROR] Indexing failed: {exc}", flush=True)

    worker = threading.Thread(target=_index_worker, daemon=True)
    worker.start()

    return {
        "started": True,
        "directories": directories,
        "incremental": incremental,
        "message": "Indexing started in background",
    }


@app.post("/index/reset")
async def reset_index():
    """Clear in-memory index and remove persisted cache."""
    if retrieval_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    with _index_status_lock:
        if _index_status["is_indexing"]:
            raise HTTPException(status_code=409, detail="Cannot reset while indexing is in progress")

    retrieval_service.clear_index()

    cache_path = Path(_get_index_cache_path())
    semantic_cache_paths = [Path(path) for path in retrieval_service.get_semantic_cache_paths(str(cache_path))]
    cache_removed = False
    removed_semantic_cache_paths = []
    try:
        if cache_path.exists():
            cache_path.unlink()
            cache_removed = True
        for semantic_cache_path in semantic_cache_paths:
            if semantic_cache_path.exists():
                semantic_cache_path.unlink()
                removed_semantic_cache_paths.append(str(semantic_cache_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Index cleared but failed to remove cache: {exc}")

    with _index_status_lock:
        _index_status["is_indexing"] = False
        _index_status["indexed_files"] = 0
        _index_status["current_path"] = None
        _index_status["directories"] = []
        _index_status["started_at"] = None
        _index_status["finished_at"] = _utcnow_iso()
        _index_status["error"] = None

    return {
        "cleared": True,
        "cache_removed": cache_removed,
        "semantic_cache_removed": bool(removed_semantic_cache_paths),
        "cache_path": str(cache_path),
        "semantic_cache_paths": removed_semantic_cache_paths,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    _enforce_public_endpoint("/health")
    with _index_status_lock:
        is_indexing = _index_status["is_indexing"]
        indexed_files = _index_status["indexed_files"]

    return {
        "status": "healthy",
        "initialized": retrieval_service is not None,
        "is_indexing": is_indexing,
        "indexed_files": indexed_files,
    }


# Simplified 2-API endpoints for UI

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Simple search endpoint for the UI.
    
    Searches file metadata and returns results with relevance scores.
    """
    if retrieval_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    file_type_filter, extension_filter = _resolve_document_filters(request.document_type)

    with _index_status_lock:
        is_indexing = _index_status["is_indexing"]
        indexed_files = _index_status["indexed_files"]

    if is_indexing and indexed_files == 0:
        raise HTTPException(
            status_code=503,
            detail="Indexing in progress. Please wait for initial scan to complete.",
        )
    
    start_time = time.time()
    
    retrieval_service.reset_limits()
    
    results = retrieval_service.retrieve(
        query=request.query,
        extension_filter=extension_filter,
        file_type_filter=file_type_filter,
        max_results=request.max_results,
        needs_content=False,
        personal_profile=request.personal_profile,
        allowed_file_types_override=request.allowed_file_types,
        blocked_extensions_override=request.blocked_extensions,
    )
    
    search_time_ms = (time.time() - start_time) * 1000
    
    # Convert to simple format with normalized relevance percent for UI
    raw_scores = [r.score for r in results]
    min_score = min(raw_scores) if raw_scores else 0.0
    max_score = max(raw_scores) if raw_scores else 0.0

    def _to_relevance_percent(score: float) -> float:
        if max_score <= min_score:
            return 100.0
        return ((score - min_score) / (max_score - min_score)) * 100.0

    simple_results = []
    for r in results:
        simple_results.append(FileMetadataSimple(
            filename=r.metadata.filename,
            path=r.metadata.full_path,
            item_type="folder" if r.metadata.file_type == "folder" else "file",
            extension=r.metadata.extension,
            size=r.metadata.size_bytes,
            modified_date=r.metadata.modified_date.isoformat(),
            relevance_score=round(_to_relevance_percent(r.score), 1),
            match_reasons=r.snippets,
            score_breakdown=r.score_breakdown,
        ))
    
    debug_payload = retrieval_service.get_last_debug_info() if request.debug else None

    return SearchResponse(
        results=simple_results,
        search_time_ms=search_time_ms,
        total_found=len(simple_results),
        debug=debug_payload,
    )


@app.get("/download")
async def download(path: str = Query(..., description="File path to download")):
    """
    Download a file by path.
    
    Performs safety checks before allowing download.
    """
    if retrieval_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    file_path = Path(path)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Folder download -> zip archive
    if file_path.is_dir():
        allowed, reason = retrieval_service.policy.is_path_allowed(str(file_path))
        if not allowed:
            raise HTTPException(status_code=403, detail=f"Access denied: {reason}")

        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        temp_zip_path = Path(temp_zip.name)
        temp_zip.close()

        try:
            added_files = 0
            with zipfile.ZipFile(temp_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(file_path):
                    root_path = Path(root)
                    for name in files:
                        candidate = root_path / name

                        allowed_candidate, _ = retrieval_service.policy.is_path_allowed(str(candidate))
                        if not allowed_candidate:
                            continue

                        rel_path = candidate.relative_to(file_path)
                        try:
                            zipf.write(candidate, arcname=str(rel_path))
                            added_files += 1
                        except (OSError, PermissionError):
                            continue

            if added_files == 0:
                try:
                    temp_zip_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise HTTPException(status_code=403, detail="No allowed files in folder to zip")

            return FileResponse(
                path=str(temp_zip_path),
                filename=f"{file_path.name or 'folder'}.zip",
                media_type="application/zip",
                background=BackgroundTask(lambda: temp_zip_path.unlink(missing_ok=True)),
            )
        except HTTPException:
            raise
        except Exception as e:
            try:
                temp_zip_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise HTTPException(status_code=500, detail=f"Error creating zip: {str(e)}")

    if not file_path.is_file():
        raise HTTPException(status_code=400, detail="Path must be a file or folder")

    # File download
    can_read, reason = retrieval_service.policy.can_read_content(path, 0)
    if not can_read:
        raise HTTPException(status_code=403, detail=f"Access denied: {reason}")

    file_size = file_path.stat().st_size
    if file_size > retrieval_service.policy.max_file_size_bytes:
        raise HTTPException(
            status_code=403,
            detail=f"File too large: {file_size} bytes (max: {retrieval_service.policy.max_file_size_bytes})"
        )

    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="application/octet-stream",
    )


@app.get("/index/status", response_model=IndexStatusResponse)
async def index_status():
    """Return startup indexing status for UI progress display."""
    _enforce_public_endpoint("/index/status")
    with _index_status_lock:
        return IndexStatusResponse(**_index_status)
