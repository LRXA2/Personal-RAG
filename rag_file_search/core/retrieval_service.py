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
import pickle
import sqlite3
import json
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

    # Whole-index lexical retrieval. This anchors semantic retrieval to exact
    # names/path terms and reduces semantic collisions.
    enable_bm25_retrieval: bool = True
    bm25_candidate_multiplier: int = 3
    rrf_k: int = 60
    lexical_candidate_weight: float = 1.0
    bm25_candidate_weight: float = 1.2
    semantic_candidate_weight: float = 0.8

    # Multi-term queries should not be dominated by candidates matching only one
    # exact anchor (for example a name without the requested document concept).
    enable_query_coverage_gate: bool = True
    min_query_coverage: float = 0.5
    query_coverage_gate_min_tokens: int = 3

    # Optional semantic reranking (metadata embeddings only)
    enable_semantic_reranking: bool = True
    semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_trust_remote_code: bool = False
    semantic_query_prompt_name: Optional[str] = None
    semantic_document_prompt_name: Optional[str] = None
    lexical_weight: float = 0.7
    semantic_weight: float = 0.3
    semantic_candidate_multiplier: int = 3
    semantic_index_batch_size: int = 64
    enable_two_phase_semantic_search: bool = True
    semantic_candidate_pool_multiplier: int = 6
    semantic_fallback_min_coverage: float = 0.67

    # Query-aware semantic type priors. These softly nudge broad queries toward
    # matching file categories without hard-coding query words.
    enable_semantic_type_priors: bool = True
    semantic_type_prior_weight: float = 12.0
    semantic_store_backend: str = "sqlite"

    # If a folder ranks highly, surface its best child files nearby.
    enable_top_folder_expansion: bool = True
    top_folders_to_expand: int = 5
    max_child_results_per_folder: int = 5
    folder_child_score_decay: float = 0.92
    max_folder_summary_children: int = 30

    # Optional cross-encoder reranking of top hybrid candidates
    enable_cross_encoder_reranking: bool = False
    reranker_model_name: Optional[str] = None
    reranker_trust_remote_code: bool = False
    reranker_top_k: int = 50
    reranker_weight: float = 0.25

    # Optional local LLM query planner. Retrieval remains deterministic and
    # falls back to the rule-based planner if model loading or JSON parsing fails.
    enable_llm_query_planner: bool = False
    query_planner_model_name: str = "Qwen/Qwen3.5-4B"
    query_planner_trust_remote_code: bool = True
    query_planner_max_new_tokens: int = 512
    query_planner_temperature: float = 0.0

    # Stabilization toggles for broad personal-document queries.
    enable_personal_intent_prefilter: bool = True
    enable_expansion_queries: bool = False
    disable_top_folder_expansion_for_generic_personal: bool = True
    personal_profile_default: str = "balanced"


@dataclass
class MetadataSearchDoc:
    """Cached retrieval text and token statistics for one metadata item."""

    metadata: FileMetadata
    text_lower: str
    token_counts: Counter
    doc_len: int


@dataclass
class QueryPlan:
    """Structured query intent used to keep anchors separate from concepts."""

    raw_query: str
    normalized_query: str
    anchors: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    expansion_queries: list[str] = field(default_factory=list)
    preferred_extensions: list[str] = field(default_factory=list)
    disfavored_extensions: list[str] = field(default_factory=list)
    preferred_file_types: list[str] = field(default_factory=list)
    disfavored_file_types: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


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
        
        # Track how many files we've read this session
        self._files_read_count = 0
        self._chars_extracted = 0

        # Semantic reranking state (lazy-loaded)
        self._semantic_model = None
        self._semantic_model_load_attempted = False
        self._reranker_model = None
        self._reranker_model_load_attempted = False
        self._query_planner_tokenizer = None
        self._query_planner_model = None
        self._query_planner_load_attempted = False
        self._query_planner_prompt_template: Optional[str] = None
        self._embedding_cache: dict[str, tuple[str, list[float]]] = {}
        self._last_semantic_scores: dict[str, float] = {}
        self._last_score_breakdowns: dict[str, dict[str, float]] = {}
        self._type_profile_embeddings: dict[str, list[float]] = {}
        self._type_profile_model_key: Optional[str] = None
        self._last_type_prior_scores: dict[str, float] = {}
        self._folder_expanded_from: dict[str, str] = {}
        self._folder_children_cache: Optional[dict[str, list[FileMetadata]]] = None
        self._directory_summary_cache: dict[str, str] = {}
        self._search_doc_cache: dict[str, MetadataSearchDoc] = {}
        self._query_embedding_cache: dict[tuple[str, str], list[float]] = {}
        self._last_debug_info: dict = {}
        self._last_gate_stats: dict = {}
    
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
        self._invalidate_retrieval_caches(clear_query_embeddings=False)
        self._rebuild_semantic_index()
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

        self._invalidate_retrieval_caches(clear_query_embeddings=False)
        self._rebuild_semantic_index()
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
        personal_profile: Optional[str] = None,
        allowed_file_types_override: Optional[list[str]] = None,
        blocked_extensions_override: Optional[list[str]] = None,
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
        query_plan = self._plan_query(query)
        search_query = query_plan.normalized_query
        query_tokens = self._tokenize(search_query)
        generic_personal_query = self._is_generic_personal_query(query_tokens)
        prefilter_profile = self._build_prefilter_profile(
            query_tokens,
            query_plan,
            extension_filter,
            file_type_filter,
            personal_profile=personal_profile,
            allowed_file_types_override=allowed_file_types_override,
            blocked_extensions_override=blocked_extensions_override,
        )

        effective_extension_filter = extension_filter
        effective_file_type_filter = file_type_filter

        # If query is broad personal-doc intent and caller did not specify filters,
        # narrow candidates away from developer artifacts.
        if generic_personal_query and not extension_filter and not file_type_filter:
            effective_file_type_filter = ["document", "image", "text"]

        # Stage 1: Metadata retrieval
        self._last_semantic_scores = {}
        self._last_score_breakdowns = {}
        self._folder_expanded_from = {}
        self._last_gate_stats = {}

        metadata_results = self.indexer.search(
            query=search_query,
            extension_filter=effective_extension_filter,
            file_type_filter=effective_file_type_filter,
            date_from=date_from,
            date_to=date_to,
            folder_contains=folder_contains,
            max_results=self.config.max_metadata_results,
        )
        metadata_results = self._apply_prefilter_to_scored(metadata_results, prefilter_profile)
        lexical_count = len(metadata_results)

        bm25_results = self._bm25_search_all(
            query=search_query,
            query_plan=query_plan,
            extension_filter=effective_extension_filter,
            file_type_filter=effective_file_type_filter,
            date_from=date_from,
            date_to=date_to,
            folder_contains=folder_contains,
            max_results=self.config.max_metadata_results,
            prefilter_profile=prefilter_profile,
        )
        bm25_count = len(bm25_results)
        semantic_candidate_pool = self._semantic_candidate_pool(
            metadata_results,
            bm25_results,
            max_results=self.config.max_metadata_results,
        )
        semantic_results = self._semantic_search_all(
            query=search_query,
            extension_filter=effective_extension_filter,
            file_type_filter=effective_file_type_filter,
            date_from=date_from,
            date_to=date_to,
            folder_contains=folder_contains,
            max_results=self.config.max_metadata_results,
            candidate_pool=semantic_candidate_pool,
            prefilter_profile=prefilter_profile,
        )
        semantic_results = self._maybe_expand_semantic_search(
            query=search_query,
            current_semantic_results=semantic_results,
            extension_filter=effective_extension_filter,
            file_type_filter=effective_file_type_filter,
            date_from=date_from,
            date_to=date_to,
            folder_contains=folder_contains,
            max_results=self.config.max_metadata_results,
            prefilter_profile=prefilter_profile,
        )
        semantic_count = len(semantic_results)
        metadata_results = self._merge_ranked_candidate_results(
            [
                (metadata_results, self.config.lexical_candidate_weight),
                (bm25_results, self.config.bm25_candidate_weight),
                (semantic_results, self.config.semantic_candidate_weight),
            ]
        )
        metadata_results = self._apply_query_coverage_gate(search_query, metadata_results, query_plan)

        if self.config.enable_reranking:
            metadata_results = self._rerank_metadata_results(search_query, metadata_results, query_plan)

        metadata_results = self._cross_encoder_rerank(search_query, metadata_results)
        can_expand_top_folders = not (
            self.config.disable_top_folder_expansion_for_generic_personal and generic_personal_query
        )
        if can_expand_top_folders:
            metadata_results = self._expand_top_folder_results(
                search_query,
                metadata_results,
                extension_filter=effective_extension_filter,
                file_type_filter=effective_file_type_filter,
                date_from=date_from,
                date_to=date_to,
                folder_contains=folder_contains,
                prefilter_profile=prefilter_profile,
            )

        self._last_debug_info = {
            "query": query,
            "search_query": search_query,
            "query_plan": {
                "anchors": query_plan.anchors,
                "concepts": query_plan.concepts,
                "expansion_queries": query_plan.expansion_queries,
                "preferred_extensions": query_plan.preferred_extensions,
                "disfavored_extensions": query_plan.disfavored_extensions,
                "preferred_file_types": query_plan.preferred_file_types,
                "disfavored_file_types": query_plan.disfavored_file_types,
                "notes": query_plan.notes,
            },
            "prefilter_profile": prefilter_profile,
            "stage_counts": {
                "lexical": lexical_count,
                "bm25": bm25_count,
                "semantic": semantic_count,
                "final": len(metadata_results),
            },
            "top_folder_expansion_applied": can_expand_top_folders,
            "gate_stats": self._last_gate_stats,
        }

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

    def _normalize_personal_profile(self, value: Optional[str]) -> str:
        profile = (value or self.config.personal_profile_default or "balanced").strip().lower()
        return profile if profile in {"strict", "balanced", "off"} else "balanced"

    def _normalize_file_types(self, values: Optional[list[str]]) -> list[str]:
        allowed = {"folder", "text", "code", "document", "data", "image", "video", "other"}
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in values or []:
            if not isinstance(item, str):
                continue
            token = item.strip().lower()
            if token in allowed and token not in seen:
                seen.add(token)
                cleaned.append(token)
        return cleaned

    def _prepare_search_query(self, query: str) -> str:
        """Normalize query for ranking to reduce generic-token noise."""
        tokens = self._tokenize(query)
        if not tokens:
            return query

        return " ".join(tokens)

    def _plan_query(self, query: str) -> QueryPlan:
        """Create a structured plan. This can be replaced by an LLM planner later."""
        llm_plan = self._plan_query_with_llm(query)
        if llm_plan is not None:
            return llm_plan

        return self._plan_query_rule_based(query)

    def _plan_query_rule_based(self, query: str) -> QueryPlan:
        """Create a structured plan using lightweight deterministic rules."""
        normalized_query = self._prepare_search_query(query)
        tokens = self._tokenize(normalized_query)

        anchors, concepts = self._split_anchor_and_concept_tokens(tokens)

        # If every token looks specific, keep the full query as the concept too.
        if not concepts and len(tokens) > 1:
            concepts = tokens[:]

        preferred_types = []
        disfavored_types = []
        preferred_extensions: list[str] = []
        disfavored_extensions: list[str] = []
        concept_text = " ".join(concepts)
        if concept_text:
            preferred_types = self._plan_preferred_file_types(concept_text)
            preferred_extensions, disfavored_extensions = self._plan_preferred_extensions(concept_text, preferred_types)
            if any(t in preferred_types for t in {"document", "image", "text"}):
                disfavored_types = ["code"]

        return QueryPlan(
            raw_query=query,
            normalized_query=normalized_query,
            anchors=anchors,
            concepts=concepts,
            expansion_queries=[],
            preferred_extensions=preferred_extensions,
            disfavored_extensions=disfavored_extensions,
            preferred_file_types=preferred_types,
            disfavored_file_types=disfavored_types,
            notes=["rule_based"],
        )

    def _split_anchor_and_concept_tokens(self, tokens: list[str]) -> tuple[list[str], list[str]]:
        """Fallback split for when LLM planning is unavailable."""
        if not tokens:
            return [], []

        concept_markers = self._generic_intent_tokens()
        anchors = [token for token in tokens if token not in concept_markers]
        concepts = [token for token in tokens if token in concept_markers]

        if not anchors:
            anchors = tokens[:1]
        if not concepts:
            concepts = [token for token in tokens if token not in anchors]

        return anchors, concepts

    def _generic_intent_tokens(self) -> set[str]:
        """Small neutral intent vocabulary for rule-based fallback planner."""
        tokens = {
            "file", "files", "folder", "folders", "directory", "directories",
            "find", "show", "search", "get", "locate", "where", "what",
            "latest", "recent", "today", "yesterday", "week", "month", "year",
        }
        for text in self._type_profiles().values():
            tokens.update(self._tokenize(text))
        return tokens

    def _plan_query_with_llm(self, query: str) -> Optional[QueryPlan]:
        """Use an optional local instruct model to produce a structured query plan."""
        if not self.config.enable_llm_query_planner:
            return None

        model_bundle = self._get_query_planner_model()
        if model_bundle is None:
            return None

        tokenizer, model = model_bundle
        prompt = self._build_query_planner_prompt(query)

        try:
            import torch
            inputs = tokenizer(prompt, return_tensors="pt")
            if hasattr(model, "device"):
                inputs = {key: value.to(model.device) for key, value in inputs.items()}

            generation_kwargs = {
                "max_new_tokens": self.config.query_planner_max_new_tokens,
                "do_sample": self.config.query_planner_temperature > 0,
                "temperature": max(self.config.query_planner_temperature, 1e-6),
                "pad_token_id": tokenizer.eos_token_id,
            }
            with torch.no_grad():
                output = model.generate(**inputs, **generation_kwargs)
            generated = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        except Exception:
            return None

        payload = self._extract_json_object(generated)
        if payload is None:
            return None

        return self._query_plan_from_payload(query, payload)

    def _get_query_planner_model(self):
        """Load the optional query planner model lazily."""
        if self._query_planner_model is not None and self._query_planner_tokenizer is not None:
            return self._query_planner_tokenizer, self._query_planner_model
        if self._query_planner_load_attempted:
            return None

        self._query_planner_load_attempted = True
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self.config.query_planner_model_name,
                trust_remote_code=self.config.query_planner_trust_remote_code,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.config.query_planner_model_name,
                trust_remote_code=self.config.query_planner_trust_remote_code,
                torch_dtype="auto",
                device_map="auto" if torch.cuda.is_available() else None,
            )
            model.eval()
            self._query_planner_tokenizer = tokenizer
            self._query_planner_model = model
            return tokenizer, model
        except Exception:
            self._query_planner_tokenizer = None
            self._query_planner_model = None
            return None

    def _build_query_planner_prompt(self, query: str) -> str:
        template = self._get_query_planner_prompt_template()
        escaped_query = query.replace('"', '\\"')
        available_extensions = sorted(list(self.indexer.get_all_extensions()))
        ext_text = ", ".join(available_extensions[:120]) if available_extensions else ""
        return (
            template
            .replace("{query}", escaped_query)
            .replace("{available_extensions}", ext_text)
        )

    def _get_query_planner_prompt_template(self) -> str:
        if self._query_planner_prompt_template is not None:
            return self._query_planner_prompt_template

        prompt_path = Path(__file__).parent / "prompts" / "query_planner_prompt.txt"
        try:
            self._query_planner_prompt_template = prompt_path.read_text(encoding="utf-8")
        except Exception:
            self._query_planner_prompt_template = 'User query: "{query}"\nJSON:'
        return self._query_planner_prompt_template

    def _extract_json_object(self, text: str) -> Optional[dict]:
        start = text.find("{")
        if start < 0:
            return None

        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            char = text[idx]
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        payload = json.loads(text[start:idx + 1])
                    except Exception:
                        return None
                    return payload if isinstance(payload, dict) else None
        return None

    def _query_plan_from_payload(self, query: str, payload: dict) -> Optional[QueryPlan]:
        allowed_file_types = {"folder", "document", "image", "text", "code", "notebook", "data", "video", "archive", "other"}

        anchors = self._clean_plan_strings(payload.get("anchors", []))
        concepts_raw = self._clean_plan_strings(payload.get("concepts", []))
        concept_single = payload.get("concept")
        if isinstance(concept_single, str) and concept_single.strip():
            concepts_raw.append(concept_single.strip().lower())
        concepts = self._tokenize(" ".join(concepts_raw))
        preferred_ext = self._normalize_extensions(payload.get("preferred_extensions", []))
        disfavored_ext = self._normalize_extensions(payload.get("disfavored_extensions", []))
        preferred = [item for item in self._clean_plan_strings(payload.get("preferred_file_types", [])) if item in allowed_file_types]
        disfavored = [item for item in self._clean_plan_strings(payload.get("disfavored_file_types", [])) if item in allowed_file_types]

        expansion_queries: list[str] = []
        if self.config.enable_expansion_queries:
            expansion_queries = self._filter_expansion_queries(
                self._clean_plan_strings(payload.get("expansion_queries", [])),
                anchors,
            )

        normalized_query = self._prepare_search_query(" ".join([query] + expansion_queries))
        if not normalized_query:
            normalized_query = self._prepare_search_query(query)

        if not anchors and not concepts:
            return None

        return QueryPlan(
            raw_query=query,
            normalized_query=normalized_query,
            anchors=anchors,
            concepts=concepts,
            expansion_queries=expansion_queries,
            preferred_extensions=preferred_ext,
            disfavored_extensions=disfavored_ext,
            preferred_file_types=preferred,
            disfavored_file_types=disfavored,
            notes=["llm"],
        )

    def _clean_plan_strings(self, value) -> list[str]:
        if not isinstance(value, list):
            return []
        cleaned = []
        for item in value:
            if not isinstance(item, str):
                continue
            text = item.strip().lower()
            if text:
                cleaned.append(text)
        return cleaned

    def _normalize_extensions(self, value) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        if not isinstance(value, list):
            return normalized
        for item in value:
            if not isinstance(item, str):
                continue
            ext = item.strip().lower()
            if not ext:
                continue
            if not ext.startswith("."):
                ext = f".{ext}"
            if ext in seen:
                continue
            seen.add(ext)
            normalized.append(ext)
        return normalized

    def _filter_expansion_queries(self, expansions: list[str], anchors: list[str]) -> list[str]:
        if not expansions:
            return []
        filtered: list[str] = []
        for expansion in expansions:
            text = expansion.strip().lower()
            if not text:
                continue
            if anchors and not all(anchor in text for anchor in anchors):
                continue
            filtered.append(text)
            if len(filtered) >= 2:
                break
        return filtered

    def _plan_preferred_file_types(self, concept_text: str) -> list[str]:
        concept_lower = concept_text.lower()
        if self._contains_any(concept_lower, {"project", "code", "script", "notebook"}):
            return ["folder", "code", "notebook"]
        if self._contains_any(concept_lower, {"image", "photo", "scan"}):
            return ["image", "document"]
        if self._contains_any(concept_lower, {"data", "spreadsheet"}):
            return ["data", "document"]
        if self._contains_any(concept_lower, {"personal", "information", "document", "documentation", "record", "passport", "identity", "bank", "tax", "insurance", "statement"}):
            return ["document", "image", "text", "folder"]
        return []

    def _plan_preferred_extensions(self, concept_text: str, preferred_file_types: list[str]) -> tuple[list[str], list[str]]:
        concept_lower = concept_text.lower()
        preferred: list[str] = []
        disfavored: list[str] = []

        if self._contains_any(concept_lower, {"project", "code", "script", "notebook"}) or any(
            file_type in {"code", "notebook"} for file_type in preferred_file_types
        ):
            preferred.extend([".py", ".ipynb", ".js", ".ts", ".txt", ".md"])
            disfavored.extend([".jpg", ".jpeg", ".png", ".mp4", ".mov", ".mkv"])
        elif any(file_type in {"document", "image", "text"} for file_type in preferred_file_types):
            preferred.extend([".pdf", ".docx", ".doc", ".txt", ".md", ".png", ".jpg", ".jpeg"])
            disfavored.extend([".py", ".js", ".ts", ".json", ".yaml", ".yml"])

        return self._dedupe_extensions(preferred), self._dedupe_extensions(disfavored)

    def _dedupe_extensions(self, extensions: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for ext in extensions:
            value = ext.strip().lower()
            if not value:
                continue
            if not value.startswith("."):
                value = f".{value}"
            if value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return normalized

    def _rerank_metadata_results(
        self,
        query: str,
        metadata_results: list[tuple[FileMetadata, float]],
        query_plan: Optional[QueryPlan] = None,
    ) -> list[tuple[FileMetadata, float]]:
        """Rerank metadata candidates using BM25-style lexical scoring + heuristics."""
        if not metadata_results:
            return metadata_results

        query_terms = self._expand_query_terms(query, query_plan)
        query_tokens = [term for term, _ in query_terms]
        query_token_weights = dict(query_terms)
        if not query_tokens:
            return metadata_results

        docs: list[tuple[FileMetadata, float, Counter, int, str, str]] = []
        doc_freq: Counter = Counter()

        for metadata, base_score in metadata_results:
            doc = self._get_search_doc(metadata)
            docs.append((metadata, base_score, doc.token_counts, doc.doc_len, doc.text_lower, metadata.filename.lower()))

            for token in doc.token_counts.keys():
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
        type_priors = self._semantic_type_priors(query)
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
            type_prior_bonus = self._metadata_type_prior_bonus(metadata, type_priors)
            feature_bonus += type_prior_bonus
            coverage_bonus = self._query_coverage_bonus(query_tokens, doc_text_lower)
            feature_bonus += coverage_bonus
            plan_bonus = self._query_plan_bonus(query_plan, metadata, doc_text_lower)
            feature_bonus += plan_bonus

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
                breakdown = {
                    "base": round((base_component / total_component_abs) * 100.0, 1),
                    "lexical": round((bm25_component / total_component_abs) * 100.0, 1),
                    "intent": round((feature_component / total_component_abs) * 100.0, 1),
                    "recency": round((recency_component / total_component_abs) * 100.0, 1),
                }
                if type_prior_bonus:
                    breakdown["type_prior"] = round(type_prior_bonus, 2)
                    self._last_type_prior_scores[metadata.full_path.lower()] = type_prior_bonus
                if coverage_bonus:
                    breakdown["coverage"] = round(coverage_bonus, 2)
                if plan_bonus:
                    breakdown["query_plan"] = round(plan_bonus, 2)
                self._last_score_breakdowns[metadata.full_path.lower()] = breakdown

        reranked.sort(key=lambda x: x[1], reverse=True)
        reranked = self._apply_result_diversity(reranked)

        if self.config.enable_semantic_reranking:
            reranked = self._semantic_rerank(query, reranked)

        return reranked

    def _bm25_search_all(
        self,
        query: str,
        query_plan: Optional[QueryPlan] = None,
        extension_filter: Optional[list[str]] = None,
        file_type_filter: Optional[list[str]] = None,
        date_from=None,
        date_to=None,
        folder_contains: Optional[str] = None,
        max_results: int = 50,
        prefilter_profile: Optional[dict] = None,
    ) -> list[tuple[FileMetadata, float]]:
        """Retrieve candidates from a BM25 pass across all filtered metadata."""
        if not self.config.enable_bm25_retrieval:
            return []

        query_terms = self._expand_query_terms(query, query_plan)
        query_tokens = [term for term, _ in query_terms]
        query_token_weights = dict(query_terms)
        if not query_tokens:
            return []

        docs: list[MetadataSearchDoc] = []
        doc_freq: Counter = Counter()
        for metadata in self.indexer.index:
            if not self._metadata_matches_filters(
                metadata,
                extension_filter=extension_filter,
                file_type_filter=file_type_filter,
                date_from=date_from,
                date_to=date_to,
                folder_contains=folder_contains,
                prefilter_profile=prefilter_profile,
            ):
                continue

            doc = self._get_search_doc(metadata)
            docs.append(doc)
            for token in doc.token_counts.keys():
                doc_freq[token] += 1

        if not docs:
            return []

        doc_count = len(docs)
        avg_doc_len = sum(max(doc.doc_len, 1) for doc in docs) / doc_count
        query_lower = query.lower().strip()
        k1 = 1.2
        b = 0.75
        scored: list[tuple[FileMetadata, float]] = []

        for doc in docs:
            score = 0.0
            for token in query_tokens:
                tf = doc.token_counts.get(token, 0)
                if tf == 0:
                    continue

                df = doc_freq.get(token, 0)
                idf = math.log(((doc_count - df + 0.5) / (df + 0.5)) + 1.0)
                numerator = tf * (k1 + 1.0)
                denominator = tf + k1 * (1.0 - b + b * (doc.doc_len / max(avg_doc_len, 1e-6)))
                score += query_token_weights.get(token, 1.0) * idf * (numerator / max(denominator, 1e-6))

            if query_lower and query_lower in doc.text_lower:
                score += 2.0

            if score > 0:
                scored.append((doc.metadata, score * 20.0))

        scored.sort(key=lambda item: item[1], reverse=True)
        limit = max(max_results * max(self.config.bm25_candidate_multiplier, 1), max_results)
        return scored[:limit]

    def _semantic_search_all(
        self,
        query: str,
        extension_filter: Optional[list[str]] = None,
        file_type_filter: Optional[list[str]] = None,
        date_from=None,
        date_to=None,
        folder_contains: Optional[str] = None,
        max_results: int = 50,
        candidate_pool: Optional[list[FileMetadata]] = None,
        force_full_index: bool = False,
        prefilter_profile: Optional[dict] = None,
    ) -> list[tuple[FileMetadata, float]]:
        """Retrieve candidates from metadata embeddings across the whole index."""
        if not self.config.enable_semantic_reranking or not query.strip():
            return []

        model = self._get_semantic_model()
        if model is None:
            return []

        source_items = candidate_pool if candidate_pool is not None and not force_full_index else self.indexer.index
        candidates = [
            metadata for metadata in source_items
            if self._metadata_matches_filters(
                metadata,
                extension_filter=extension_filter,
                file_type_filter=file_type_filter,
                date_from=date_from,
                date_to=date_to,
                folder_contains=folder_contains,
                prefilter_profile=prefilter_profile,
            )
        ]
        if not candidates:
            return []

        self._ensure_semantic_embeddings(candidates)

        try:
            query_embedding = self._get_query_embedding(model, query)
        except Exception:
            return []

        scored: list[tuple[FileMetadata, float]] = []
        for metadata in candidates:
            cache_entry = self._embedding_cache.get(metadata.full_path.lower())
            if not cache_entry:
                continue

            semantic_similarity = sum(a * b for a, b in zip(query_embedding, cache_entry[1]))
            semantic_score = ((semantic_similarity + 1.0) / 2.0) * 100.0
            scored.append((metadata, semantic_score))
            self._last_semantic_scores[metadata.full_path.lower()] = semantic_score / 100.0

        scored.sort(key=lambda item: item[1], reverse=True)
        limit = max(max_results * max(self.config.semantic_candidate_multiplier, 1), max_results)
        return scored[:limit]

    def _merge_candidate_results(
        self,
        lexical_results: list[tuple[FileMetadata, float]],
        semantic_results: list[tuple[FileMetadata, float]],
    ) -> list[tuple[FileMetadata, float]]:
        """Merge lexical and vector candidates, preserving the best score per path."""
        if not semantic_results:
            return lexical_results

        by_path: dict[str, tuple[FileMetadata, float]] = {}
        for metadata, score in lexical_results + semantic_results:
            key = metadata.full_path.lower()
            existing = by_path.get(key)
            if existing is None or score > existing[1]:
                by_path[key] = (metadata, score)

        merged = list(by_path.values())
        merged.sort(key=lambda item: item[1], reverse=True)
        limit = max(
            self.config.max_metadata_results * max(self.config.semantic_candidate_multiplier, 1),
            self.config.max_metadata_results,
        )
        return merged[:limit]

    def _merge_ranked_candidate_results(
        self,
        result_sets: list[tuple[list[tuple[FileMetadata, float]], float]],
    ) -> list[tuple[FileMetadata, float]]:
        """Fuse ranked candidate lists using weighted reciprocal-rank fusion."""
        fused: dict[str, tuple[FileMetadata, float]] = {}
        rrf_k = max(self.config.rrf_k, 1)

        for results, weight in result_sets:
            if not results or weight <= 0:
                continue

            for rank, (metadata, _) in enumerate(results, start=1):
                key = metadata.full_path.lower()
                contribution = weight / (rrf_k + rank)
                if key in fused:
                    existing_metadata, existing_score = fused[key]
                    fused[key] = (existing_metadata, existing_score + contribution)
                else:
                    fused[key] = (metadata, contribution)

        if not fused:
            return []

        merged = [(metadata, score * 1000.0) for metadata, score in fused.values()]
        merged.sort(key=lambda item: item[1], reverse=True)
        limit = max(
            self.config.max_metadata_results * max(
                self.config.semantic_candidate_multiplier,
                self.config.bm25_candidate_multiplier,
                1,
            ),
            self.config.max_metadata_results,
        )
        return merged[:limit]

    def _semantic_candidate_pool(
        self,
        lexical_results: list[tuple[FileMetadata, float]],
        bm25_results: list[tuple[FileMetadata, float]],
        max_results: int,
    ) -> Optional[list[FileMetadata]]:
        """Build a broad candidate pool for the semantic pass."""
        if not self.config.enable_two_phase_semantic_search:
            return None

        limit = max(max_results * max(self.config.semantic_candidate_pool_multiplier, 1), max_results)
        by_path: dict[str, FileMetadata] = {}
        for results in (lexical_results, bm25_results):
            for metadata, _ in results:
                by_path.setdefault(metadata.full_path.lower(), metadata)
                if len(by_path) >= limit:
                    break
            if len(by_path) >= limit:
                break

        if not by_path:
            return None
        return list(by_path.values())

    def _maybe_expand_semantic_search(
        self,
        query: str,
        current_semantic_results: list[tuple[FileMetadata, float]],
        extension_filter: Optional[list[str]] = None,
        file_type_filter: Optional[list[str]] = None,
        date_from=None,
        date_to=None,
        folder_contains: Optional[str] = None,
        max_results: int = 50,
        prefilter_profile: Optional[dict] = None,
    ) -> list[tuple[FileMetadata, float]]:
        """Fallback to full-index semantic search when fast-pool coverage is weak."""
        if not self.config.enable_two_phase_semantic_search or not self.config.enable_semantic_reranking:
            return current_semantic_results

        query_tokens = self._coverage_query_tokens(query)
        if len(query_tokens) < self.config.query_coverage_gate_min_tokens:
            return current_semantic_results

        best_coverage = 0.0
        for metadata, _ in current_semantic_results[:10]:
            coverage = self._query_coverage_ratio(query_tokens, self._get_search_doc(metadata).text_lower)
            best_coverage = max(best_coverage, coverage)

        if best_coverage >= self.config.semantic_fallback_min_coverage:
            return current_semantic_results

        fallback_results = self._semantic_search_all(
            query=query,
            extension_filter=extension_filter,
            file_type_filter=file_type_filter,
            date_from=date_from,
            date_to=date_to,
            folder_contains=folder_contains,
            max_results=max_results,
            candidate_pool=None,
            force_full_index=True,
            prefilter_profile=prefilter_profile,
        )
        return self._merge_candidate_results(current_semantic_results, fallback_results)

    def _apply_query_coverage_gate(
        self,
        query: str,
        metadata_results: list[tuple[FileMetadata, float]],
        query_plan: Optional[QueryPlan] = None,
    ) -> list[tuple[FileMetadata, float]]:
        """Suppress one-anchor collisions when better-covered candidates exist."""
        if not self.config.enable_query_coverage_gate or not metadata_results:
            return metadata_results

        self._last_gate_stats = {
            "input_candidates": len(metadata_results),
            "anchor_gate_applied": False,
            "anchor_matches": 0,
            "coverage_gate_applied": False,
            "coverage_threshold": self.config.min_query_coverage,
            "coverage_matches": 0,
        }

        anchor_filtered = self._apply_anchor_gate(metadata_results, query_plan)
        if anchor_filtered is not None:
            metadata_results = anchor_filtered
            self._last_gate_stats["anchor_gate_applied"] = True
            self._last_gate_stats["anchor_matches"] = len(anchor_filtered)

        query_tokens = self._coverage_query_tokens(query)
        if len(query_tokens) < self.config.query_coverage_gate_min_tokens:
            self._last_gate_stats["coverage_reason"] = "short_query"
            return metadata_results

        scored: list[tuple[FileMetadata, float, float]] = []
        for metadata, score in metadata_results:
            coverage = self._query_coverage_ratio(query_tokens, self._get_search_doc(metadata).text_lower)
            scored.append((metadata, score, coverage))

        filtered = [
            (metadata, score) for metadata, score, coverage in scored
            if coverage >= self.config.min_query_coverage
        ]
        self._last_gate_stats["coverage_gate_applied"] = True
        self._last_gate_stats["coverage_matches"] = len(filtered)

        # Keep recall when every candidate is weakly covered.
        if not filtered:
            self._last_gate_stats["coverage_reason"] = "fallback_keep_recall"
            return metadata_results

        self._last_gate_stats["coverage_reason"] = "filtered"
        return filtered

    def _apply_anchor_gate(
        self,
        metadata_results: list[tuple[FileMetadata, float]],
        query_plan: Optional[QueryPlan],
    ) -> Optional[list[tuple[FileMetadata, float]]]:
        """Require entity/path anchors when enough anchored candidates exist."""
        if not query_plan or not query_plan.anchors:
            return None

        anchored = [
            (metadata, score) for metadata, score in metadata_results
            if self._matches_query_anchors(query_plan, self._get_search_doc(metadata).text_lower)
        ]
        # Fallback-friendly behavior: only apply anchor gate when we have matches.
        return anchored if anchored else None

    def _matches_query_anchors(self, query_plan: QueryPlan, doc_text_lower: str) -> bool:
        if not query_plan.anchors:
            return True
        return all(anchor in doc_text_lower for anchor in query_plan.anchors)

    def _coverage_query_tokens(self, query: str) -> list[str]:
        return [
            token for token in self._tokenize(query)
            if token not in {"file", "files", "folder", "folders", "find", "show", "search"}
        ]

    def _query_coverage_ratio(self, query_tokens: list[str], doc_text_lower: str) -> float:
        if not query_tokens:
            return 0.0

        matched = 0
        for token in query_tokens:
            if token in doc_text_lower:
                matched += 1
        return matched / len(query_tokens)

    def _query_plan_bonus(
        self,
        query_plan: Optional[QueryPlan],
        metadata: FileMetadata,
        doc_text_lower: str,
    ) -> float:
        if not query_plan:
            return 0.0

        bonus = 0.0
        if query_plan.anchors:
            if self._matches_query_anchors(query_plan, doc_text_lower):
                bonus += 14.0
            else:
                bonus -= 40.0

        if query_plan.concepts:
            concept_coverage = self._query_coverage_ratio(query_plan.concepts, doc_text_lower)
            if concept_coverage >= 0.5:
                bonus += 8.0
            elif len(query_plan.concepts) > 0:
                bonus -= 8.0

        profile = self._metadata_type_profile(metadata)
        if profile in query_plan.preferred_file_types:
            bonus += 8.0
        if profile in query_plan.disfavored_file_types:
            bonus -= 18.0

        ext = metadata.extension.lower()
        if ext and ext in query_plan.preferred_extensions:
            bonus += 9.0
        if ext and ext in query_plan.disfavored_extensions:
            bonus -= 20.0

        return bonus

    def _metadata_matches_filters(
        self,
        metadata: FileMetadata,
        extension_filter: Optional[list[str]] = None,
        file_type_filter: Optional[list[str]] = None,
        date_from=None,
        date_to=None,
        folder_contains: Optional[str] = None,
        prefilter_profile: Optional[dict] = None,
    ) -> bool:
        if extension_filter and metadata.extension not in extension_filter:
            return False
        if file_type_filter and metadata.file_type not in file_type_filter:
            return False
        if date_from and metadata.modified_date < date_from:
            return False
        if date_to and metadata.modified_date > date_to:
            return False
        if folder_contains and folder_contains.lower() not in metadata.parent_folder.lower():
            return False
        if prefilter_profile and not self._passes_prefilter_profile(metadata, prefilter_profile):
            return False
        return True

    def _apply_prefilter_to_scored(
        self,
        metadata_results: list[tuple[FileMetadata, float]],
        prefilter_profile: Optional[dict],
    ) -> list[tuple[FileMetadata, float]]:
        if not prefilter_profile or not metadata_results:
            return metadata_results
        return [
            (metadata, score)
            for metadata, score in metadata_results
            if self._passes_prefilter_profile(metadata, prefilter_profile)
        ]

    def _build_prefilter_profile(
        self,
        query_tokens: list[str],
        query_plan: Optional[QueryPlan],
        extension_filter: Optional[list[str]],
        file_type_filter: Optional[list[str]],
        personal_profile: Optional[str] = None,
        allowed_file_types_override: Optional[list[str]] = None,
        blocked_extensions_override: Optional[list[str]] = None,
    ) -> Optional[dict]:
        if not self.config.enable_personal_intent_prefilter:
            return None
        if extension_filter or file_type_filter:
            return None

        profile = self._normalize_personal_profile(personal_profile)
        if profile == "off":
            return None

        has_personal_intent = self._is_personal_query(query_tokens)
        has_document_intent = any(token in {"document", "documents", "doc", "docs", "record", "records"} for token in query_tokens)
        prefers_docs = bool(query_plan and any(t in {"document", "image", "text"} for t in query_plan.preferred_file_types))
        if not (has_personal_intent or has_document_intent or prefers_docs):
            return None

        is_navigation = self._is_navigation_query(query_tokens)
        generic_personal_tokens = {"personal", "important", "private", "information", "details", "document", "documents", "doc", "docs"}
        focused_tokens = [token for token in query_tokens if token not in generic_personal_tokens]
        broad_personal_query = has_personal_intent and len(focused_tokens) <= 1 and not is_navigation

        allowed_file_types = ["document", "image", "text", "folder"]
        if broad_personal_query:
            # For broad personal queries, folder hits are often noisy (e.g. generic
            # project/UI folders containing the user's home path token).
            allowed_file_types = ["document", "image", "text"]
        blocked_extensions = [
            ".py", ".pyi", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c", ".h", ".hpp",
            ".json", ".yaml", ".yml", ".toml", ".xml", ".lock", ".log",
        ]
        if profile == "strict":
            blocked_extensions.extend([".md", ".rst", ".ini", ".cfg"])

        override_types = self._normalize_file_types(allowed_file_types_override)
        if override_types:
            allowed_file_types = override_types

        override_exts = self._normalize_extensions(blocked_extensions_override)
        if override_exts:
            blocked_extensions = override_exts

        return {
            "profile": profile,
            "allowed_file_types": allowed_file_types,
            "blocked_extensions": blocked_extensions,
            "blocked_path_markers": [
                "\\venv\\", "\\site-packages\\", "\\.git\\", "\\node_modules\\", "\\__pycache__\\",
                "\\dist-info\\", "\\build\\", "\\dist\\", "\\cache\\", "\\tmp\\", "\\temp\\",
            ],
        }

    def _passes_prefilter_profile(self, metadata: FileMetadata, prefilter_profile: dict) -> bool:
        allowed_file_types = set(prefilter_profile.get("allowed_file_types") or [])
        blocked_extensions = set(prefilter_profile.get("blocked_extensions") or [])
        blocked_path_markers = set(prefilter_profile.get("blocked_path_markers") or [])

        if allowed_file_types and metadata.file_type not in allowed_file_types:
            return False
        if metadata.extension.lower() in blocked_extensions:
            return False

        full_path_lower = metadata.full_path.lower()
        parent_lower = metadata.parent_folder.lower()
        for marker in blocked_path_markers:
            marker_lower = marker.lower()
            if marker_lower in full_path_lower or marker_lower in parent_lower:
                return False
        return True

    def _get_search_doc(self, metadata: FileMetadata) -> MetadataSearchDoc:
        key = metadata.full_path.lower()
        cached = self._search_doc_cache.get(key)
        if cached is not None:
            return cached

        text_lower = self._build_rerank_text(metadata).lower()
        token_counts = Counter(self._tokenize(text_lower))
        doc = MetadataSearchDoc(
            metadata=metadata,
            text_lower=text_lower,
            token_counts=token_counts,
            doc_len=sum(token_counts.values()),
        )
        self._search_doc_cache[key] = doc
        return doc

    def _semantic_type_priors(self, query: str) -> dict[str, float]:
        """Return soft query-fit scores for general file categories."""
        if not self.config.enable_semantic_type_priors or not query.strip():
            return {}

        model = self._get_semantic_model()
        if model is None:
            return {}

        self._ensure_type_profile_embeddings(model)
        if not self._type_profile_embeddings:
            return {}

        try:
            query_embedding = self._get_query_embedding(model, query)
        except Exception:
            return {}

        scores: dict[str, float] = {}
        for profile_name, profile_embedding in self._type_profile_embeddings.items():
            similarity = sum(a * b for a, b in zip(query_embedding, profile_embedding))
            scores[profile_name] = (similarity + 1.0) / 2.0

        return scores

    def _ensure_type_profile_embeddings(self, model) -> None:
        """Build reusable embeddings for general file-type descriptions."""
        model_key = self._semantic_model_cache_key()
        if self._type_profile_embeddings and self._type_profile_model_key == model_key:
            return

        profiles = self._type_profiles()
        try:
            encoded = self._encode_documents(model, list(profiles.values()))
        except Exception:
            self._type_profile_embeddings = {}
            self._type_profile_model_key = None
            return

        self._type_profile_embeddings = {
            profile_name: self._embedding_to_list(embedding)
            for profile_name, embedding in zip(profiles.keys(), encoded)
        }
        self._type_profile_model_key = model_key

    def _type_profiles(self) -> dict[str, str]:
        """General category descriptions used for semantic type matching."""
        return {
            "folder": "folder directory project workspace collection grouped files repository location",
            "code": "source code programming script software module API implementation development",
            "notebook": "jupyter notebook analysis experiment data science machine learning research code",
            "document": "document report pdf word file written information form record documentation",
            "image": "image photo picture scan screenshot scanned document visual file",
            "data": "data spreadsheet csv table structured records json yaml configuration dataset",
            "video": "video recording movie clip footage presentation media",
            "text": "text notes markdown readme plain text documentation instructions",
            "archive": "archive zip compressed backup package bundle collection",
        }

    def _metadata_type_prior_bonus(self, metadata: FileMetadata, type_priors: dict[str, float]) -> float:
        """Map semantic type-fit scores to a small score adjustment."""
        if not type_priors:
            return 0.0

        profile = self._metadata_type_profile(metadata)
        score = type_priors.get(profile)
        if score is None:
            return 0.0

        centered_score = score - self._average_type_prior(type_priors)
        return centered_score * self.config.semantic_type_prior_weight

    def _metadata_type_profile(self, metadata: FileMetadata) -> str:
        ext = metadata.extension.lower()
        if ext in {".ipynb"}:
            return "notebook"
        if ext in {".zip", ".rar", ".7z", ".tar", ".gz"}:
            return "archive"
        if metadata.file_type in {"folder", "code", "document", "image", "data", "video", "text"}:
            return metadata.file_type
        return "document" if ext in {".pdf", ".doc", ".docx", ".rtf", ".odt"} else "text"

    def _average_type_prior(self, type_priors: dict[str, float]) -> float:
        if not type_priors:
            return 0.0
        return sum(type_priors.values()) / len(type_priors)

    def _invalidate_retrieval_caches(self, clear_query_embeddings: bool = False) -> None:
        """Clear caches derived from current metadata/index structure."""
        self._folder_children_cache = None
        self._directory_summary_cache = {}
        self._search_doc_cache = {}
        self._type_profile_embeddings = {}
        self._type_profile_model_key = None
        if clear_query_embeddings:
            self._query_embedding_cache = {}

    def _rebuild_semantic_index(self) -> None:
        """Build metadata embeddings for the current index when the model is available."""
        if not self.config.enable_semantic_reranking or not self.indexer.index:
            return

        valid_paths = {metadata.full_path.lower() for metadata in self.indexer.index}
        stale_paths = [path for path in self._embedding_cache.keys() if path not in valid_paths]
        for path in stale_paths:
            self._embedding_cache.pop(path, None)

        self._ensure_semantic_embeddings(self.indexer.index)

    def _ensure_semantic_embeddings(self, metadata_items: list[FileMetadata]) -> None:
        """Ensure metadata embeddings exist for the provided items."""
        model = self._get_semantic_model()
        if model is None:
            return

        texts: list[str] = []
        keys: list[str] = []
        signatures: list[str] = []

        for metadata in metadata_items:
            key = metadata.full_path.lower()
            semantic_text = self._build_semantic_text(metadata)
            signature = self._embedding_signature(metadata, semantic_text)
            cache_entry = self._embedding_cache.get(key)
            if cache_entry and cache_entry[0] == signature:
                continue

            keys.append(key)
            texts.append(semantic_text)
            signatures.append(signature)

        batch_size = max(self.config.semantic_index_batch_size, 1)
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            try:
                encoded = self._encode_documents(model, batch_texts)
            except Exception:
                return

            for idx, embedding in enumerate(encoded):
                absolute_idx = start + idx
                self._embedding_cache[keys[absolute_idx]] = (
                    signatures[absolute_idx],
                    self._embedding_to_list(embedding),
                )

    def _embedding_signature(self, metadata: FileMetadata, semantic_text: str) -> str:
        return f"{metadata.modified_date.isoformat()}|{metadata.size_bytes}|{semantic_text}"

    def _embedding_to_list(self, embedding) -> list[float]:
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return list(embedding)

    def _get_query_embedding(self, model, query: str) -> list[float]:
        cache_key = (self._semantic_model_cache_key(), query.strip().lower())
        cached = self._query_embedding_cache.get(cache_key)
        if cached is not None:
            return cached

        embedding = self._embedding_to_list(self._encode_query(model, query)[0])
        self._query_embedding_cache[cache_key] = embedding
        return embedding

    def _encode_query(self, model, query: str):
        kwargs = {"normalize_embeddings": True}
        if self.config.semantic_query_prompt_name:
            kwargs["prompt_name"] = self.config.semantic_query_prompt_name
        try:
            return model.encode([query], **kwargs)
        except TypeError:
            kwargs.pop("prompt_name", None)
            return model.encode([query], **kwargs)

    def _encode_documents(self, model, texts: list[str]):
        kwargs = {"normalize_embeddings": True}
        if self.config.semantic_document_prompt_name:
            kwargs["prompt_name"] = self.config.semantic_document_prompt_name
        try:
            return model.encode(texts, **kwargs)
        except TypeError:
            kwargs.pop("prompt_name", None)
            return model.encode(texts, **kwargs)

    def _cross_encoder_rerank(
        self,
        query: str,
        metadata_results: list[tuple[FileMetadata, float]],
    ) -> list[tuple[FileMetadata, float]]:
        """Optionally rerank top candidates with a local cross-encoder reranker."""
        if not self.config.enable_cross_encoder_reranking or not self.config.reranker_model_name:
            return metadata_results

        reranker = self._get_reranker_model()
        if reranker is None or not metadata_results:
            return metadata_results

        top_k = max(1, min(self.config.reranker_top_k, len(metadata_results)))
        top_results = metadata_results[:top_k]
        remaining = metadata_results[top_k:]
        pairs = [(query, self._build_semantic_text(metadata)) for metadata, _ in top_results]

        try:
            raw_scores = reranker.predict(pairs)
        except Exception:
            return metadata_results

        scores = self._scores_to_list(raw_scores)
        if len(scores) != len(top_results):
            return metadata_results

        lexical_scores = [score for _, score in top_results]
        min_lex = min(lexical_scores)
        max_lex = max(lexical_scores)
        min_rerank = min(scores)
        max_rerank = max(scores)
        weight = max(0.0, min(1.0, self.config.reranker_weight))

        reranked: list[tuple[FileMetadata, float]] = []
        for (metadata, lexical_score), rerank_score in zip(top_results, scores):
            lexical_norm = 1.0 if max_lex <= min_lex else (lexical_score - min_lex) / (max_lex - min_lex)
            rerank_norm = 1.0 if max_rerank <= min_rerank else (rerank_score - min_rerank) / (max_rerank - min_rerank)
            combined_score = ((1.0 - weight) * lexical_norm + weight * rerank_norm) * 100.0
            reranked.append((metadata, combined_score))

            self._last_score_breakdowns[metadata.full_path.lower()] = {
                "hybrid": round((1.0 - weight) * 100.0, 1),
                "reranker": round(weight * 100.0, 1),
            }

        reranked.sort(key=lambda item: item[1], reverse=True)
        return reranked + remaining

    def _scores_to_list(self, scores) -> list[float]:
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        return [float(score) for score in scores]

    def _expand_top_folder_results(
        self,
        query: str,
        metadata_results: list[tuple[FileMetadata, float]],
        extension_filter: Optional[list[str]] = None,
        file_type_filter: Optional[list[str]] = None,
        date_from=None,
        date_to=None,
        folder_contains: Optional[str] = None,
        prefilter_profile: Optional[dict] = None,
    ) -> list[tuple[FileMetadata, float]]:
        """Add relevant child files for the highest-ranked folder results."""
        if not self.config.enable_top_folder_expansion or not metadata_results:
            return metadata_results

        existing_paths = {metadata.full_path.lower() for metadata, _ in metadata_results}
        expanded_results = list(metadata_results)
        folders = [
            (metadata, score) for metadata, score in metadata_results
            if metadata.file_type == "folder"
        ][:max(self.config.top_folders_to_expand, 0)]

        for folder_metadata, folder_score in folders:
            child_results = self._rank_child_files_for_folder(
                query,
                folder_metadata,
                folder_score,
                extension_filter=extension_filter,
                file_type_filter=file_type_filter,
                date_from=date_from,
                date_to=date_to,
                folder_contains=folder_contains,
                prefilter_profile=prefilter_profile,
            )
            for child_metadata, child_score in child_results:
                child_key = child_metadata.full_path.lower()
                if child_key in existing_paths:
                    continue
                existing_paths.add(child_key)
                expanded_results.append((child_metadata, child_score))
                self._folder_expanded_from[child_key] = folder_metadata.full_path

        expanded_results.sort(key=lambda item: item[1], reverse=True)
        return expanded_results

    def _rank_child_files_for_folder(
        self,
        query: str,
        folder_metadata: FileMetadata,
        folder_score: float,
        extension_filter: Optional[list[str]] = None,
        file_type_filter: Optional[list[str]] = None,
        date_from=None,
        date_to=None,
        folder_contains: Optional[str] = None,
        prefilter_profile: Optional[dict] = None,
    ) -> list[tuple[FileMetadata, float]]:
        """Rank child files under a matched folder using metadata and semantic signals."""
        child_candidates = [
            metadata for metadata in self._children_for_folder(folder_metadata.full_path, recursive=True)
            if metadata.file_type != "folder"
            and self._metadata_matches_filters(
                metadata,
                extension_filter=extension_filter,
                file_type_filter=file_type_filter,
                date_from=date_from,
                date_to=date_to,
                folder_contains=folder_contains,
                prefilter_profile=prefilter_profile,
            )
        ]
        if not child_candidates:
            return []

        if self.config.enable_semantic_reranking:
            self._ensure_semantic_embeddings(child_candidates)

        scored_children: list[tuple[FileMetadata, float]] = []
        query_tokens = self._tokenize(query)
        type_priors = self._semantic_type_priors(query)
        for child in child_candidates:
            score = folder_score * self.config.folder_child_score_decay
            score += self._child_lexical_bonus(query_tokens, child)
            score += self._metadata_type_prior_bonus(child, type_priors)
            score += self._child_semantic_bonus(query, child)
            score -= self._folder_depth_penalty(child.full_path, folder_metadata.full_path)
            scored_children.append((child, score))

        scored_children.sort(key=lambda item: item[1], reverse=True)
        return scored_children[:max(self.config.max_child_results_per_folder, 0)]

    def _child_lexical_bonus(self, query_tokens: list[str], metadata: FileMetadata) -> float:
        if not query_tokens:
            return 0.0

        filename_lower = metadata.filename.lower()
        path_context_lower = self._build_path_context(metadata).lower()
        bonus = 0.0
        for token in query_tokens:
            if token in filename_lower:
                bonus += 8.0
            elif token in path_context_lower:
                bonus += 3.0
        return bonus

    def _child_semantic_bonus(self, query: str, metadata: FileMetadata) -> float:
        if not self.config.enable_semantic_reranking:
            return 0.0

        semantic_score = self._last_semantic_scores.get(metadata.full_path.lower())
        if semantic_score is None:
            model = self._get_semantic_model()
            cache_entry = self._embedding_cache.get(metadata.full_path.lower())
            if model is None or cache_entry is None:
                return 0.0
            try:
                query_embedding = self._get_query_embedding(model, query)
            except Exception:
                return 0.0
            similarity = sum(a * b for a, b in zip(query_embedding, cache_entry[1]))
            semantic_score = (similarity + 1.0) / 2.0
            self._last_semantic_scores[metadata.full_path.lower()] = semantic_score

        return max(0.0, semantic_score - 0.5) * 12.0

    def _folder_depth_penalty(self, file_path: str, folder_path: str) -> float:
        child_parts = self._normalized_path_parts(file_path)
        folder_parts = self._normalized_path_parts(folder_path)
        depth = max(len(child_parts) - len(folder_parts) - 1, 0)
        return min(depth * 2.0, 10.0)

    def _is_under_folder(self, file_path: str, folder_path: str) -> bool:
        file_norm = self._normalized_path_text(file_path)
        folder_norm = self._normalized_path_text(folder_path).rstrip("/")
        return file_norm.startswith(folder_norm + "/")

    def _children_for_folder(self, folder_path: str, recursive: bool = False) -> list[FileMetadata]:
        if recursive:
            folder_norm = self._normalized_path_text(folder_path).rstrip("/")
            return [
                metadata for metadata in self.indexer.index
                if self._normalized_path_text(metadata.full_path) != folder_norm
                and self._is_under_folder(metadata.full_path, folder_path)
            ]

        cache = self._folder_children_cache
        if cache is None:
            cache = self._build_folder_children_cache()
            self._folder_children_cache = cache

        return cache.get(self._normalized_path_text(folder_path).rstrip("/"), [])

    def _build_folder_children_cache(self) -> dict[str, list[FileMetadata]]:
        cache: dict[str, list[FileMetadata]] = defaultdict(list)
        for metadata in self.indexer.index:
            parent_key = self._normalized_path_text(metadata.parent_folder).rstrip("/")
            cache[parent_key].append(metadata)

        for children in cache.values():
            children.sort(key=lambda item: (item.file_type == "folder", item.modified_date), reverse=True)

        return cache

    def _normalized_path_text(self, path: str) -> str:
        return str(path).replace("\\", "/").lower().rstrip("/")

    def _normalized_path_parts(self, path: str) -> list[str]:
        return [part for part in self._normalized_path_text(path).split("/") if part]

    def _get_reranker_model(self):
        """Load optional cross-encoder reranker lazily; return None if unavailable."""
        if self._reranker_model is not None:
            return self._reranker_model
        if self._reranker_model_load_attempted:
            return None

        self._reranker_model_load_attempted = True
        try:
            from sentence_transformers import CrossEncoder
            self._reranker_model = CrossEncoder(
                self.config.reranker_model_name,
                trust_remote_code=self.config.reranker_trust_remote_code,
            )
            return self._reranker_model
        except Exception:
            self._reranker_model = None
            return None

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
            query_embedding = self._get_query_embedding(model, query)

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
                encoded = self._encode_documents(model, doc_texts)
                idx = 0
                for i, emb in enumerate(embeddings):
                    if emb is None:
                        vector = self._embedding_to_list(encoded[idx])
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
            self._semantic_model = SentenceTransformer(
                self.config.semantic_model_name,
                trust_remote_code=self.config.semantic_trust_remote_code,
            )
            return self._semantic_model
        except Exception:
            self._semantic_model = None
            return None

    def _build_semantic_text(self, metadata: FileMetadata) -> str:
        """Build semantic text representation for metadata embeddings."""
        path_context = self._build_path_context(metadata)
        directory_summary = self._build_directory_summary(metadata)
        parts = [
            f"name {metadata.filename}",
            f"type {metadata.file_type} {metadata.extension}".strip(),
            path_context,
            directory_summary,
        ]
        if metadata.file_type == "folder":
            parts.append("directory folder path")
        return " ".join([p for p in parts if p])

    def _build_rerank_text(self, metadata: FileMetadata) -> str:
        """Build searchable metadata text for reranking."""
        path_context = self._build_path_context(metadata)
        directory_summary = self._build_directory_summary(metadata)
        return " ".join([
            metadata.filename,
            path_context,
            directory_summary,
            metadata.extension,
            metadata.file_type,
        ])

    def _build_directory_summary(self, metadata: FileMetadata) -> str:
        """Summarize nearby child names so folders can match their contents."""
        cache_key = (
            f"folder:{self._normalized_path_text(metadata.full_path)}"
            if metadata.file_type == "folder"
            else f"parent:{self._normalized_path_text(metadata.parent_folder)}"
        )
        cached = self._directory_summary_cache.get(cache_key)
        if cached is not None:
            return cached

        if metadata.file_type == "folder":
            children = self._children_for_folder(metadata.full_path)
            label = "folder contains"
        else:
            children = self._children_for_folder(metadata.parent_folder)
            label = "parent folder contains"

        if not children:
            self._directory_summary_cache[cache_key] = ""
            return ""

        children = children[:max(self.config.max_folder_summary_children, 0)]
        child_parts = []
        for child in children:
            name_text = " ".join(self._tokenize(Path(child.filename).stem.replace("_", " ").replace("-", " ")))
            type_text = child.file_type
            ext_text = child.extension.lstrip(".")
            child_parts.append(" ".join([part for part in [name_text, type_text, ext_text] if part]))

        summary = " ".join([part for part in child_parts if part])
        result = f"{label} {summary}" if summary else ""
        self._directory_summary_cache[cache_key] = result
        return result

    def _build_path_context(self, metadata: FileMetadata) -> str:
        """Build deterministic path breadcrumbs for lexical and semantic matching."""
        path_parts = self._meaningful_path_parts(metadata.full_path)
        parent_parts = self._meaningful_path_parts(metadata.parent_folder)

        nearby = parent_parts[-5:]
        root_context = parent_parts[:3]
        breadcrumb = path_parts[-7:]
        direct_parent = parent_parts[-1:] if parent_parts else []

        parts = [
            f"parent {' '.join(direct_parent)}" if direct_parent else "",
            f"near folders {' '.join(nearby)}" if nearby else "",
            f"path breadcrumbs {' '.join(breadcrumb)}" if breadcrumb else "",
            f"root context {' '.join(root_context)}" if root_context else "",
        ]
        return " ".join([part for part in parts if part])

    def _meaningful_path_parts(self, path: str) -> list[str]:
        """Return cleaned path parts while dropping common low-signal containers."""
        noisy_parts = {
            "", "/", "\\", "users", "user", "documents", "downloads", "desktop",
            "onedrive", "dropbox", "google drive", "icloud drive", "program files",
            "program files (x86)", "appdata", "local", "roaming", "temp", "tmp",
        }

        meaningful: list[str] = []
        for raw_part in Path(path).parts:
            part = raw_part.strip().strip("\\/")
            if not part or part.endswith(":"):
                continue

            normalized = part.lower()
            if normalized in noisy_parts:
                continue

            cleaned = " ".join(self._tokenize(part.replace("_", " ").replace("-", " ")))
            if cleaned:
                meaningful.append(cleaned)

        return meaningful

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

    def _expand_query_terms(self, query: str, query_plan: Optional[QueryPlan] = None) -> list[tuple[str, float]]:
        """Build weighted terms from structured plan (anchors/concepts/expansions)."""
        seen: dict[str, float] = {}

        def _add_terms(text: str, weight: float):
            for token in self._tokenize(text):
                seen[token] = max(seen.get(token, 0.0), weight)

        _add_terms(query, 1.0)

        if query_plan is not None:
            if query_plan.anchors:
                _add_terms(" ".join(query_plan.anchors), 1.35)
            if query_plan.concepts:
                _add_terms(" ".join(query_plan.concepts), 1.05)
            for expansion in query_plan.expansion_queries:
                _add_terms(expansion, 0.9)

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

    def _query_coverage_bonus(self, query_tokens: list[str], doc_text_lower: str) -> float:
        """Reward candidates that cover more of the query and penalize one-token collisions."""
        meaningful_tokens = [token for token in query_tokens if token not in {"file", "files", "folder", "folders"}]
        if len(meaningful_tokens) < 2:
            return 0.0

        coverage = self._query_coverage_ratio(meaningful_tokens, doc_text_lower)
        if coverage >= 0.9:
            return 12.0
        if coverage >= 0.67:
            return 5.0
        if coverage <= 0.34:
            return -18.0
        return -6.0

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

        expanded_from = self._folder_expanded_from.get(metadata.full_path.lower())
        if expanded_from:
            reasons.append(f"inside matched folder: {Path(expanded_from).name}")

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

    def get_last_debug_info(self) -> dict:
        """Return debug data from the most recent retrieval call."""
        return dict(self._last_debug_info)

    def save_index_cache(self, cache_path: str, scanned_directories: Optional[list[str]] = None) -> None:
        """Persist index to disk for faster startup."""
        self.indexer.save_cache(cache_path, scanned_directories=scanned_directories)
        self._save_semantic_cache(cache_path)

    def load_index_cache(self, cache_path: str) -> tuple[int, list[str]]:
        """Load index from disk cache."""
        loaded = self.indexer.load_cache(cache_path)
        self._invalidate_retrieval_caches(clear_query_embeddings=False)
        if loaded[0] > 0:
            self._load_semantic_cache(cache_path)
            self._rebuild_semantic_index()
        return loaded

    def clear_index(self) -> None:
        """Clear all indexed metadata from memory."""
        self.indexer.index = []
        self.indexer._path_to_metadata = {}
        self._embedding_cache = {}
        self._last_semantic_scores = {}
        self._last_score_breakdowns = {}
        self._invalidate_retrieval_caches(clear_query_embeddings=True)
        self.reset_limits()

    def get_semantic_cache_path(self, cache_path: str) -> str:
        """Return the sidecar path used for persisted metadata embeddings."""
        return str(self._semantic_cache_path(cache_path))

    def get_semantic_cache_paths(self, cache_path: str) -> list[str]:
        """Return all known sidecar paths for persisted metadata embeddings."""
        path = Path(cache_path)
        return [
            str(path.with_name(f"{path.stem}.embeddings.sqlite3")),
            str(path.with_name(f"{path.stem}.embeddings{path.suffix}")),
        ]

    def _semantic_cache_path(self, cache_path: str) -> Path:
        path = Path(cache_path)
        if self.config.semantic_store_backend == "sqlite":
            return path.with_name(f"{path.stem}.embeddings.sqlite3")
        return path.with_name(f"{path.stem}.embeddings{path.suffix}")

    def _save_semantic_cache(self, cache_path: str) -> None:
        """Persist metadata embeddings separately from the metadata index."""
        if not self._embedding_cache:
            return

        if self.config.semantic_store_backend == "sqlite":
            self._save_semantic_cache_sqlite(cache_path)
            return

        path = self._semantic_cache_path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "model_key": self._semantic_model_cache_key(),
            "items": self._embedding_cache,
        }

        try:
            with open(path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    def _load_semantic_cache(self, cache_path: str) -> None:
        """Load persisted metadata embeddings when they match the configured model."""
        if self.config.semantic_store_backend == "sqlite":
            self._load_semantic_cache_sqlite(cache_path)
            return

        path = self._semantic_cache_path(cache_path)
        if not path.exists():
            return

        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except Exception:
            return

        model_key = payload.get("model_key") or payload.get("model_name")
        if model_key != self._semantic_model_cache_key():
            return

        items = payload.get("items", {})
        if isinstance(items, dict):
            self._embedding_cache = items

    def _save_semantic_cache_sqlite(self, cache_path: str) -> None:
        """Persist metadata embeddings to a local SQLite store."""
        path = self._semantic_cache_path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model_key = self._semantic_model_cache_key()

        try:
            with sqlite3.connect(path) as conn:
                self._init_semantic_store(conn)
                conn.execute("DELETE FROM embeddings WHERE model_key != ?", (model_key,))
                rows = [
                    (path_key, model_key, signature, sqlite3.Binary(pickle.dumps(vector, protocol=pickle.HIGHEST_PROTOCOL)))
                    for path_key, (signature, vector) in self._embedding_cache.items()
                ]
                conn.executemany(
                    """
                    INSERT INTO embeddings(path_key, model_key, signature, vector)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(path_key, model_key) DO UPDATE SET
                        signature = excluded.signature,
                        vector = excluded.vector
                    """,
                    rows,
                )
                conn.commit()
        except Exception:
            pass

    def _load_semantic_cache_sqlite(self, cache_path: str) -> None:
        """Load metadata embeddings from a local SQLite store."""
        path = self._semantic_cache_path(cache_path)
        if not path.exists():
            return

        model_key = self._semantic_model_cache_key()
        try:
            with sqlite3.connect(path) as conn:
                self._init_semantic_store(conn)
                rows = conn.execute(
                    "SELECT path_key, signature, vector FROM embeddings WHERE model_key = ?",
                    (model_key,),
                ).fetchall()
        except Exception:
            return

        loaded: dict[str, tuple[str, list[float]]] = {}
        for path_key, signature, vector_blob in rows:
            try:
                loaded[path_key] = (signature, pickle.loads(vector_blob))
            except Exception:
                continue

        if loaded:
            self._embedding_cache = loaded

    def _init_semantic_store(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                path_key TEXT NOT NULL,
                model_key TEXT NOT NULL,
                signature TEXT NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY (path_key, model_key)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_key)"
        )

    def _semantic_model_cache_key(self) -> str:
        return "|".join([
            self.config.semantic_model_name,
            self.config.semantic_query_prompt_name or "",
            self.config.semantic_document_prompt_name or "",
        ])
