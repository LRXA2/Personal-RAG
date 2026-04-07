"""
Two-stage retrieval service.

This module implements the core retrieval pipeline:
1. Stage 1: File-level retrieval using metadata search
2. Stage 2: Content-level retrieval for shortlisted files
"""

from typing import Optional
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import math
import re
from datetime import datetime
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

    # Candidate reranking
    enable_reranking: bool = True

    # Optional semantic reranking (metadata embeddings only)
    enable_semantic_reranking: bool = True
    semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    lexical_weight: float = 0.7
    semantic_weight: float = 0.3


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
        self._stop_tokens = {
            "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "by", "and", "or", "with", "from", "that", "this", "it", "as", "be", "where",
        }
        self._query_synonyms = {
            "passport": ["identity", "id", "travel", "visa"],
            "id": ["identity", "passport", "license"],
            "resume": ["cv"],
            "cv": ["resume"],
            "tax": ["return", "irs", "finance"],
            "bank": ["statement", "account", "finance"],
            "insurance": ["policy", "claim"],
            "certificate": ["cert", "record"],
            "personal": ["private", "identity"],
            "document": ["record", "paper"],
            "documents": ["records", "papers"],
        }
        
        # Track how many files we've read this session
        self._files_read_count = 0
        self._chars_extracted = 0

        # Semantic reranking state (lazy-loaded)
        self._semantic_model = None
        self._semantic_model_load_attempted = False
        self._embedding_cache: dict[str, tuple[str, list[float]]] = {}
        self._last_semantic_scores: dict[str, float] = {}
        self._last_score_breakdowns: dict[str, dict[str, float]] = {}
    
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

    def refresh_directories(self, directories: list[str], progress_callback=None) -> dict:
        """Incrementally refresh indexed metadata for directories."""
        totals = {
            "scanned": 0,
            "added": 0,
            "updated": 0,
            "removed": 0,
        }

        running_scanned = 0
        for directory in directories:
            callback = None
            if progress_callback is not None:
                def callback(current_count, current_path, running_total=running_scanned):
                    progress_callback(running_total + current_count, current_path)

            stats = self.indexer.refresh_directory(directory, callback)
            running_scanned += stats.get("scanned", 0)

            totals["scanned"] += stats.get("scanned", 0)
            totals["added"] += stats.get("added", 0)
            totals["updated"] += stats.get("updated", 0)
            totals["removed"] += stats.get("removed", 0)

        return totals
    
    def retrieve(
        self,
        query: str,
        extension_filter: Optional[list[str]] = None,
        file_type_filter: Optional[list[str]] = None,
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
        search_query = self._prepare_search_query(query)
        query_tokens = self._tokenize(query)
        generic_personal_query = self._is_generic_personal_query(query_tokens)

        effective_extension_filter = extension_filter
        effective_file_type_filter = file_type_filter

        # If query is broad personal-doc intent and caller did not specify filters,
        # narrow candidates away from developer artifacts.
        if generic_personal_query and not extension_filter and not file_type_filter:
            effective_file_type_filter = ["document", "image", "text"]

        # Stage 1: Metadata retrieval
        self._last_semantic_scores = {}
        self._last_score_breakdowns = {}

        metadata_results = self.indexer.search(
            query=search_query,
            extension_filter=effective_extension_filter,
            file_type_filter=effective_file_type_filter,
            date_from=date_from,
            date_to=date_to,
            folder_contains=folder_contains,
            max_results=self.config.max_metadata_results,
        )

        if self.config.enable_reranking:
            metadata_results = self._rerank_metadata_results(search_query, metadata_results)

        if generic_personal_query:
            metadata_results = [(m, s) for (m, s) in metadata_results if s > 0]

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
                reasons = self._build_metadata_reasons(query, metadata)
                score_breakdown = self._last_score_breakdowns.get(metadata.full_path.lower(), {})
                results.append(SearchResult(
                    metadata=metadata,
                    score=score,
                    match_type="metadata_only",
                    snippets=reasons,
                    score_breakdown=score_breakdown,
                ))
        
        # Stage 2: Content grounding for shortlisted files
        for metadata, base_score in files_to_read:
            content_results = self._ground_in_content(query, metadata, base_score)
            if content_results:
                results.extend(content_results)
            else:
                # Fall back to metadata-only if content extraction failed
                reasons = self._build_metadata_reasons(query, metadata)
                reasons.append("content grounding unavailable")
                score_breakdown = self._last_score_breakdowns.get(metadata.full_path.lower(), {})
                results.append(SearchResult(
                    metadata=metadata,
                    score=base_score * 0.8,  # Slight penalty for no content
                    match_type="metadata_only",
                    snippets=reasons,
                    score_breakdown=score_breakdown,
                ))
        
        # Sort all results by score
        results.sort(key=lambda r: r.score, reverse=True)
        
        return results

    def _prepare_search_query(self, query: str) -> str:
        """Normalize query for ranking to reduce generic-token noise."""
        tokens = self._tokenize(query)
        if not tokens:
            return query

        if self._is_personal_query(tokens):
            generic = {"personal", "important", "document", "documents", "doc", "docs", "documentation"}
            focused = [t for t in tokens if t not in generic]
            if focused:
                return " ".join(focused)

        return " ".join(tokens)

    def _rerank_metadata_results(
        self,
        query: str,
        metadata_results: list[tuple[FileMetadata, float]],
    ) -> list[tuple[FileMetadata, float]]:
        """Rerank metadata candidates using BM25-style lexical scoring + heuristics."""
        if not metadata_results:
            return metadata_results

        query_terms = self._expand_query_terms(query)
        query_tokens = [term for term, _ in query_terms]
        query_token_weights = dict(query_terms)
        if not query_tokens:
            return metadata_results

        docs: list[tuple[FileMetadata, float, Counter, int, str, str]] = []
        doc_freq: Counter = Counter()

        for metadata, base_score in metadata_results:
            doc_text = self._build_rerank_text(metadata)
            tokens = self._tokenize(doc_text)
            token_counts = Counter(tokens)
            doc_len = len(tokens)
            docs.append((metadata, base_score, token_counts, doc_len, doc_text.lower(), metadata.filename.lower()))

            for token in set(tokens):
                doc_freq[token] += 1

        doc_count = len(docs)
        avg_doc_len = sum(max(d[3], 1) for d in docs) / max(doc_count, 1)

        # BM25 constants
        k1 = 1.2
        b = 0.75

        is_navigation_query = self._is_navigation_query(query_tokens)
        is_content_query = self._is_content_query(query_tokens)
        wants_recent = self._wants_recent(query_tokens)
        is_personal_query = self._is_personal_query(query_tokens)
        extension_priors = self._infer_extension_priors(query_tokens)
        query_lower = query.lower().strip()
        now = datetime.now()

        reranked: list[tuple[FileMetadata, float]] = []
        for metadata, base_score, token_counts, doc_len, doc_text_lower, filename_lower in docs:
            bm25_score = 0.0

            for token in query_tokens:
                tf = token_counts.get(token, 0)
                if tf == 0:
                    continue

                df = doc_freq.get(token, 0)
                idf = math.log(((doc_count - df + 0.5) / (df + 0.5)) + 1.0)
                numerator = tf * (k1 + 1.0)
                denominator = tf + k1 * (1.0 - b + b * (doc_len / max(avg_doc_len, 1e-6)))
                bm25_score += query_token_weights.get(token, 1.0) * idf * (numerator / max(denominator, 1e-6))

            feature_bonus = 0.0
            if query_lower:
                if filename_lower == query_lower:
                    feature_bonus += 40.0
                elif query_lower in filename_lower:
                    feature_bonus += 15.0

                if query_lower in doc_text_lower:
                    feature_bonus += 10.0

            if metadata.file_type == "folder" and is_navigation_query:
                feature_bonus += 12.0

            if metadata.file_type == "folder" and is_content_query:
                feature_bonus -= 8.0

            if metadata.file_type != "folder" and is_content_query:
                feature_bonus += 3.0

            feature_bonus += extension_priors.get(metadata.extension.lower(), 0.0)

            if is_personal_query:
                ext = metadata.extension.lower()
                parent_lower = metadata.parent_folder.lower()
                full_path_lower = metadata.full_path.lower()
                filename_lower_local = metadata.filename.lower()
                parent_tail = " ".join([p.lower() for p in Path(metadata.parent_folder).parts[-3:]])

                personal_doc_exts = {
                    ".pdf", ".doc", ".docx", ".txt", ".rtf", ".odt", ".jpg", ".jpeg", ".png",
                }
                code_like_exts = {
                    ".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs", ".proto", ".md",
                }
                technical_data_exts = {
                    ".json", ".yaml", ".yml", ".toml", ".xml", ".h", ".hpp", ".pyi",
                }
                dev_path_markers = {
                    "\\venv\\", "\\site-packages\\", "\\.git\\", "\\node_modules\\", "\\__pycache__\\",
                    "\\lib\\", "\\include\\", "\\dist-info\\", "\\build\\", "\\dist\\",
                }

                if metadata.file_type == "code" or ext in code_like_exts:
                    feature_bonus -= 95.0

                if ext in technical_data_exts:
                    feature_bonus -= 45.0

                if ext in personal_doc_exts or metadata.file_type in {"document", "image", "data"}:
                    feature_bonus += 10.0

                if self._contains_any(parent_tail, {"identity", "passport", "certificate", "finance", "tax", "bank", "insurance", "resume", "statement", "license"}):
                    feature_bonus += 16.0

                if any(marker in full_path_lower for marker in dev_path_markers):
                    feature_bonus -= 60.0

                # Penalize developer-doc style names for personal-doc queries.
                if "documentation" in filename_lower_local and (metadata.file_type == "code" or ext in code_like_exts):
                    feature_bonus -= 18.0

            recency_bonus = self._recency_bonus(metadata.modified_date, now, wants_recent)

            # Blend original ranker with reranker signals.
            base_component = base_score * 0.65
            bm25_component = bm25_score * 20.0
            feature_component = feature_bonus
            recency_component = recency_bonus

            combined_score = (base_score * 0.65) + (bm25_score * 20.0) + feature_bonus + recency_bonus
            reranked.append((metadata, combined_score))

            total_component_abs = (
                abs(base_component)
                + abs(bm25_component)
                + abs(feature_component)
                + abs(recency_component)
            )
            if total_component_abs > 0:
                self._last_score_breakdowns[metadata.full_path.lower()] = {
                    "base": round((base_component / total_component_abs) * 100.0, 1),
                    "lexical": round((bm25_component / total_component_abs) * 100.0, 1),
                    "intent": round((feature_component / total_component_abs) * 100.0, 1),
                    "recency": round((recency_component / total_component_abs) * 100.0, 1),
                }

        reranked.sort(key=lambda x: x[1], reverse=True)
        reranked = self._apply_result_diversity(reranked)

        if self.config.enable_semantic_reranking:
            reranked = self._semantic_rerank(query, reranked)

        return reranked

    def _semantic_rerank(
        self,
        query: str,
        metadata_results: list[tuple[FileMetadata, float]],
    ) -> list[tuple[FileMetadata, float]]:
        """Optionally rerank metadata candidates using sentence embeddings."""
        model = self._get_semantic_model()
        if model is None or not metadata_results:
            return metadata_results

        try:
            query_embedding = model.encode([query], normalize_embeddings=True)[0]

            doc_texts: list[str] = []
            cache_keys: list[str] = []
            embeddings: list[Optional[list[float]]] = []

            for metadata, _ in metadata_results:
                sem_text = self._build_semantic_text(metadata)
                cache_key = metadata.full_path.lower()
                signature = f"{metadata.modified_date.isoformat()}|{metadata.size_bytes}|{sem_text}"
                cache_entry = self._embedding_cache.get(cache_key)

                if cache_entry and cache_entry[0] == signature:
                    embeddings.append(cache_entry[1])
                else:
                    embeddings.append(None)
                    doc_texts.append(sem_text)
                    cache_keys.append(cache_key)

            if doc_texts:
                encoded = model.encode(doc_texts, normalize_embeddings=True)
                idx = 0
                for i, emb in enumerate(embeddings):
                    if emb is None:
                        vector = encoded[idx].tolist()
                        metadata = metadata_results[i][0]
                        sem_text = self._build_semantic_text(metadata)
                        signature = f"{metadata.modified_date.isoformat()}|{metadata.size_bytes}|{sem_text}"
                        self._embedding_cache[cache_keys[idx]] = (signature, vector)
                        embeddings[i] = vector
                        idx += 1

            lexical_scores = [score for _, score in metadata_results]
            min_score = min(lexical_scores)
            max_score = max(lexical_scores)

            lw = max(0.0, min(1.0, self.config.lexical_weight))
            sw = max(0.0, min(1.0, self.config.semantic_weight))
            total_w = lw + sw
            if total_w <= 0:
                lw, sw = 0.7, 0.3
                total_w = 1.0
            lw /= total_w
            sw /= total_w

            reranked: list[tuple[FileMetadata, float]] = []
            semantic_scores: dict[str, float] = {}
            for i, (metadata, lexical_score) in enumerate(metadata_results):
                lex_norm = 1.0
                if max_score > min_score:
                    lex_norm = (lexical_score - min_score) / (max_score - min_score)

                doc_embedding = embeddings[i]
                if doc_embedding is None:
                    sem_norm = 0.0
                else:
                    sem_sim = sum(a * b for a, b in zip(query_embedding, doc_embedding))
                    sem_norm = (sem_sim + 1.0) / 2.0

                combined_norm = (lw * lex_norm) + (sw * sem_norm)
                combined_score = combined_norm * 100.0
                reranked.append((metadata, combined_score))
                semantic_scores[metadata.full_path.lower()] = sem_norm

                lexical_component = max(0.0, lw * lex_norm)
                semantic_component = max(0.0, sw * sem_norm)
                total_component = lexical_component + semantic_component
                if total_component > 0:
                    self._last_score_breakdowns[metadata.full_path.lower()] = {
                        "lexical": round((lexical_component / total_component) * 100.0, 1),
                        "semantic": round((semantic_component / total_component) * 100.0, 1),
                    }

            reranked.sort(key=lambda x: x[1], reverse=True)
            self._last_semantic_scores = semantic_scores
            return reranked

        except Exception:
            return metadata_results

    def _get_semantic_model(self):
        """Load semantic model lazily; return None if unavailable."""
        if self._semantic_model is not None:
            return self._semantic_model
        if self._semantic_model_load_attempted:
            return None

        self._semantic_model_load_attempted = True
        try:
            from sentence_transformers import SentenceTransformer
            self._semantic_model = SentenceTransformer(self.config.semantic_model_name)
            return self._semantic_model
        except Exception:
            self._semantic_model = None
            return None

    def _build_semantic_text(self, metadata: FileMetadata) -> str:
        """Build semantic text representation for metadata embeddings."""
        parent_tail = self._path_tail_text(metadata.parent_folder, segments=3)
        parts = [
            metadata.filename,
            parent_tail,
            metadata.extension,
            metadata.file_type,
        ]
        if metadata.file_type == "folder":
            parts.append("directory folder path")
        return " ".join([p for p in parts if p])

    def _build_rerank_text(self, metadata: FileMetadata) -> str:
        """Build searchable metadata text for reranking."""
        parent_tail = self._path_tail_text(metadata.parent_folder, segments=3)
        return " ".join([
            metadata.filename,
            parent_tail,
            metadata.extension,
            metadata.file_type,
        ])

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for lexical matching."""
        if not text:
            return []
        tokens = re.findall(r"[a-z0-9_]+", text.lower())
        return [t for t in tokens if t not in self._stop_tokens and len(t) > 1]

    def _is_navigation_query(self, query_tokens: list[str]) -> bool:
        nav_tokens = {
            "folder", "directory", "path", "under", "inside", "where", "location", "parent",
        }
        return any(tok in nav_tokens for tok in query_tokens)

    def _is_content_query(self, query_tokens: list[str]) -> bool:
        content_tokens = {
            "what", "explain", "summarize", "summary", "mentions", "contains", "content", "inside", "why", "how",
        }
        return any(tok in content_tokens for tok in query_tokens)

    def _wants_recent(self, query_tokens: list[str]) -> bool:
        recent_tokens = {
            "recent", "latest", "new", "newest", "today", "yesterday", "week", "month", "updated",
        }
        return any(tok in recent_tokens for tok in query_tokens)

    def _is_personal_query(self, query_tokens: list[str]) -> bool:
        personal_tokens = {
            "personal", "passport", "identity", "id", "license", "certificate", "bank", "tax", "insurance", "statement", "resume", "cv",
        }
        important_tokens = {"important", "private"}
        document_tokens = {"document", "documents", "doc", "docs", "documentation"}

        has_personal = any(tok in personal_tokens for tok in query_tokens)
        has_important_doc_combo = (
            any(tok in important_tokens for tok in query_tokens)
            and any(tok in document_tokens for tok in query_tokens)
        )
        return has_personal or has_important_doc_combo

    def _is_generic_personal_query(self, query_tokens: list[str]) -> bool:
        """True when query asks for personal documents without specific anchor terms."""
        if not self._is_personal_query(query_tokens):
            return False

        generic = {"personal", "important", "document", "documents", "doc", "docs", "documentation"}
        focused = [t for t in query_tokens if t not in generic]
        return len(focused) == 0

    def _expand_query_terms(self, query: str) -> list[tuple[str, float]]:
        """Expand query terms with weighted synonyms."""
        tokens = self._tokenize(query)
        if not tokens:
            return []

        seen: dict[str, float] = {}
        for token in tokens:
            seen[token] = max(seen.get(token, 0.0), 1.0)
            for synonym in self._query_synonyms.get(token, []):
                seen[synonym] = max(seen.get(synonym, 0.0), 0.45)

        return list(seen.items())

    def _infer_extension_priors(self, query_tokens: list[str]) -> dict[str, float]:
        """Infer extension bonuses from query intent."""
        priors: dict[str, float] = {}

        if self._is_personal_query(query_tokens):
            for ext in [".pdf", ".docx", ".doc", ".txt", ".jpg", ".jpeg", ".png"]:
                priors[ext] = priors.get(ext, 0.0) + 12.0

        if any(tok in {"code", "python", "py", "script", "api"} for tok in query_tokens):
            for ext in [".py", ".js", ".ts", ".java", ".json"]:
                priors[ext] = priors.get(ext, 0.0) + 10.0

        if any(tok in {"image", "photo", "picture", "scan"} for tok in query_tokens):
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
                priors[ext] = priors.get(ext, 0.0) + 10.0

        return priors

    def _apply_result_diversity(
        self,
        metadata_results: list[tuple[FileMetadata, float]],
    ) -> list[tuple[FileMetadata, float]]:
        """Reduce repetitive near-duplicate results from same stems/folders."""
        if not metadata_results:
            return metadata_results

        stem_counts: defaultdict[str, int] = defaultdict(int)
        parent_counts: defaultdict[str, int] = defaultdict(int)
        diversified: list[tuple[FileMetadata, float]] = []

        for metadata, score in metadata_results:
            stem = Path(metadata.filename).stem.lower()
            parent = metadata.parent_folder.lower()

            adjusted_score = score
            if stem_counts[stem] >= 1:
                adjusted_score -= stem_counts[stem] * 12.0
            if parent_counts[parent] >= 3:
                adjusted_score -= (parent_counts[parent] - 2) * 6.0

            diversified.append((metadata, adjusted_score))
            stem_counts[stem] += 1
            parent_counts[parent] += 1

        diversified.sort(key=lambda x: x[1], reverse=True)
        return diversified

    def _contains_any(self, text: str, needles: set[str]) -> bool:
        return any(n in text for n in needles)

    def _path_tail_text(self, path: str, segments: int = 3) -> str:
        parts = [p for p in Path(path).parts if p and p not in {"/", "\\"}]
        return " ".join(parts[-segments:])

    def _recency_bonus(self, modified_date: datetime, now: datetime, wants_recent: bool) -> float:
        age_days = max((now - modified_date).days, 0)

        if age_days <= 7:
            return 10.0 if wants_recent else 5.0
        if age_days <= 30:
            return 7.0 if wants_recent else 3.0
        if age_days <= 90:
            return 4.0 if wants_recent else 1.5
        if wants_recent and age_days <= 180:
            return 1.0
        return 0.0

    def _build_metadata_reasons(self, query: str, metadata: FileMetadata) -> list[str]:
        query_tokens = self._tokenize(query)
        reason_ignore_tokens = {"personal", "important", "document", "documents", "doc", "docs", "documentation"}
        filename_lower = metadata.filename.lower()
        parent_lower = self._path_tail_text(metadata.parent_folder, segments=3).lower()
        path_lower = metadata.full_path.lower()

        reasons: list[str] = []

        name_hits = [t for t in query_tokens if t not in reason_ignore_tokens and t in filename_lower]
        if name_hits:
            reasons.append(f"name match: {', '.join(sorted(set(name_hits))[:3])}")

        folder_hits = [t for t in query_tokens if t not in reason_ignore_tokens and t in parent_lower]
        if folder_hits:
            reasons.append(f"folder match: {', '.join(sorted(set(folder_hits))[:3])}")

        if metadata.extension and any(t == metadata.extension.lstrip('.') for t in query_tokens):
            reasons.append(f"extension match: {metadata.extension}")

        if not reasons and query.strip() and query.strip().lower() in path_lower:
            reasons.append("path contains full query")

        age_days = max((datetime.now() - metadata.modified_date).days, 0)
        if age_days <= 30:
            reasons.append(f"recently modified: {age_days} day(s) ago")

        semantic_score = self._last_semantic_scores.get(metadata.full_path.lower())
        if semantic_score is not None and semantic_score >= 0.7:
            reasons.append(f"semantic match: {semantic_score:.2f}")

        if not reasons:
            if metadata.file_type == "folder":
                reasons.append("folder metadata match")
            elif metadata.extension:
                reasons.append(f"metadata match ({metadata.extension})")
            else:
                reasons.append("metadata relevance match")

        return reasons[:3]
    
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

    def save_index_cache(self, cache_path: str, scanned_directories: Optional[list[str]] = None) -> None:
        """Persist index to disk for faster startup."""
        self.indexer.save_cache(cache_path, scanned_directories=scanned_directories)

    def load_index_cache(self, cache_path: str) -> tuple[int, list[str]]:
        """Load index from disk cache."""
        return self.indexer.load_cache(cache_path)

    def clear_index(self) -> None:
        """Clear all indexed metadata from memory."""
        self.indexer.index = []
        self.indexer._path_to_metadata = {}
        self.reset_limits()
