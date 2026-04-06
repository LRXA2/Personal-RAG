"""
LLM Planner for interpreting queries and generating structured search parameters.

This module uses a local LLM to:
- Parse natural language queries
- Extract search intent (dates, file types, folders)
- Decide whether content grounding is needed
- Format final answers from retrieved evidence
"""

import re
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass


@dataclass
class SearchIntent:
    """Structured representation of search intent extracted from a query."""
    
    # The core search terms
    keywords: list[str]
    
    # File extension filter (e.g., ['.pdf', '.md'])
    extensions: Optional[list[str]] = None
    
    # Date range filters
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    
    # Folder constraints
    folder_contains: Optional[str] = None
    
    # Whether the query likely needs content reading
    needs_content: bool = False
    
    # Query type classification
    query_type: str = "file_search"  # file_search, question_answering, navigation
    
    # Raw interpretation notes
    notes: Optional[str] = None


class LLMPlanner:
    """
    Plans retrieval strategies using rule-based parsing.
    
    In v1, this uses heuristics and pattern matching.
    Can be upgraded to use a local LLM later.
    """
    
    def __init__(self):
        # Common file type keywords
        self.file_type_map = {
            "pdf": [".pdf"],
            "document": [".pdf", ".docx", ".doc", ".odt"],
            "spreadsheet": [".xlsx", ".xls", ".csv"],
            "presentation": [".pptx", ".ppt", ".odp"],
            "image": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg"],
            "video": [".mp4", ".avi", ".mkv", ".mov", ".wmv"],
            "python": [".py"],
            "py": [".py"],
            "code": [".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs"],
            "text": [".txt", ".md", ".rst"],
            "note": [".md", ".txt", ".rst"],
            "markdown": [".md"],
            "json": [".json"],
            "yaml": [".yaml", ".yml"],
            "xml": [".xml"],
            "csv": [".csv"],
        }
        
        # Date-related keywords
        self.date_keywords = {
            "today": 0,
            "yesterday": 1,
            "week": 7,
            "month": 30,
            "year": 365,
        }
    
    def parse_query(self, query: str) -> SearchIntent:
        """
        Parse a natural language query into structured search intent.
        
        Args:
            query: User's natural language query
        
        Returns:
            SearchIntent object with extracted parameters
        """
        query_lower = query.lower()
        keywords = []
        extensions = None
        date_from = None
        date_to = None
        folder_contains = None
        needs_content = False
        query_type = "file_search"
        notes = []
        
        # Extract file type / extension requests
        for file_type, exts in self.file_type_map.items():
            if file_type in query_lower:
                if extensions is None:
                    extensions = []
                extensions.extend(exts)
                notes.append(f"Detected file type: {file_type}")
        
        # Check for explicit extension mentions (e.g., "pdf files", ".txt")
        ext_matches = re.findall(r'\.(pdf|txt|md|py|js|docx|xlsx|pptx|jpg|png|mp4|mkv)', query_lower)
        if ext_matches:
            if extensions is None:
                extensions = []
            extensions.extend([f".{m}" for m in ext_matches])
        
        # Extract date constraints
        date_from, date_to, date_notes = self._extract_dates(query_lower)
        if date_notes:
            notes.extend(date_notes)
        
        # Extract folder constraints
        folder_patterns = [
            r"in\s+(?:the\s+)?(?:folder|directory)?\s*['\"]?([^'\"]+)['\"]?",
            r"(?:folder|directory)\s+(?:named?\s+)?['\"]?([^'\"]+)['\"]?",
            r"under\s+['\"]?([^'\"]+)['\"]?",
        ]
        for pattern in folder_patterns:
            match = re.search(pattern, query_lower)
            if match:
                folder_contains = match.group(1).strip()
                notes.append(f"Folder constraint: {folder_contains}")
                break
        
        # Determine if content reading is needed
        # Questions that ask about content vs. just finding files
        question_indicators = [
            "what is", "what are", "how do", "explain", "describe",
            "tell me about", "summarize", "content of", "inside",
            "says about", "mentions", "discusses",
        ]
        for indicator in question_indicators:
            if indicator in query_lower:
                needs_content = True
                query_type = "question_answering"
                notes.append("Content grounding likely needed")
                break
        
        # Extract keywords - be more inclusive
        # Keep important words including file-related terms
        stop_words = {
            "the", "a", "an", "in", "on", "at", "to", "for", "of", "with",
            "find", "show", "look", "get", "give", "me", "my",
            "i", "want", "need", "please", "can", "could", "would",
            "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did",
            "and", "or", "but", "if", "then", "else",
            "from", "by", "as", "into", "through", "during",
            "all", "any", "some", "what", "which", "where",
        }
        
        # Remove detected patterns from query for keyword extraction
        clean_query = query_lower
        for pattern in folder_patterns:
            clean_query = re.sub(pattern, " ", clean_query)
        
        # Extract remaining words as keywords - keep more terms
        words = re.findall(r'\b[a-z0-9_-]{2,}\b', clean_query)
        keywords = [w for w in words if w not in stop_words]
        
        # If no keywords extracted, use original query words
        if not keywords:
            keywords = words[:5]  # Limit to first 5
        
        return SearchIntent(
            keywords=keywords,
            extensions=extensions,
            date_from=date_from,
            date_to=date_to,
            folder_contains=folder_contains,
            needs_content=needs_content,
            query_type=query_type,
            notes="; ".join(notes) if notes else None,
        )
    
    def _extract_dates(self, query: str) -> tuple[Optional[datetime], Optional[datetime], list[str]]:
        """Extract date constraints from query."""
        date_from = None
        date_to = None
        notes = []
        today = datetime.now()
        
        # Check for relative date expressions
        if "today" in query:
            date_from = today.replace(hour=0, minute=0, second=0, microsecond=0)
            notes.append("Date filter: today")
        
        if "yesterday" in query:
            yesterday = today - timedelta(days=1)
            date_from = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            date_to = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
            notes.append("Date filter: yesterday")
        
        if "this week" in query or "last week" in query:
            if "this week" in query:
                # Start of this week (Monday)
                days_since_monday = today.weekday()
                date_from = (today - timedelta(days=days_since_monday)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                notes.append("Date filter: this week")
            else:
                # Last week
                days_since_monday = today.weekday()
                last_monday = today - timedelta(days=days_since_monday + 7)
                date_from = last_monday.replace(hour=0, minute=0, second=0, microsecond=0)
                date_to = (last_monday + timedelta(days=6)).replace(
                    hour=23, minute=59, second=59, microsecond=999999
                )
                notes.append("Date filter: last week")
        
        if "this month" in query:
            date_from = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            notes.append("Date filter: this month")
        
        if "last month" in query:
            if today.month == 1:
                date_from = today.replace(year=today.year - 1, month=12, day=1,
                                         hour=0, minute=0, second=0, microsecond=0)
            else:
                date_from = today.replace(month=today.month - 1, day=1,
                                         hour=0, minute=0, second=0, microsecond=0)
            if today.month == 1:
                date_to = today.replace(day=1, hour=23, minute=59, second=59, microsecond=999999) - timedelta(seconds=1)
            else:
                date_to = today.replace(day=1, hour=23, minute=59, second=59, microsecond=999999) - timedelta(seconds=1)
            notes.append("Date filter: last month")
        
        # Check for year mentions
        year_match = re.search(r'\b(20\d{2}|19\d{2})\b', query)
        if year_match:
            year = int(year_match.group(1))
            date_from = datetime(year, 1, 1)
            date_to = datetime(year, 12, 31, 23, 59, 59)
            notes.append(f"Date filter: year {year}")
        
        # Check for month names
        month_names = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12
        }
        for month_name, month_num in month_names.items():
            if month_name in query:
                if date_from is None:  # Don't override if already set
                    year = today.year
                    date_from = datetime(year, month_num, 1)
                    if month_num == 12:
                        date_to = datetime(year + 1, 1, 1) - timedelta(seconds=1)
                    else:
                        date_to = datetime(year, month_num + 1, 1) - timedelta(seconds=1)
                    notes.append(f"Date filter: {month_name}")
                break
        
        return date_from, date_to, notes
    
    def format_answer(
        self, 
        query: str, 
        intent: SearchIntent, 
        results: list,
    ) -> str:
        """
        Format a final answer from retrieved results.
        
        Args:
            query: Original user query
            intent: Parsed search intent
            results: List of SearchResult objects
        
        Returns:
            Formatted answer string
        """
        if not results:
            return f"No files found matching your query: '{query}'"
        
        # Group results by type
        metadata_only = [r for r in results if r.match_type == "metadata_only"]
        content_grounding = [r for r in results if r.match_type == "content_grounding"]
        
        response_parts = []
        
        # Header
        if content_grounding:
            response_parts.append(f"Found {len(results)} relevant file(s):\n")
        else:
            response_parts.append(f"Found {len(results)} file(s) matching your criteria:\n")
        
        # List top results
        for i, result in enumerate(results[:10], 1):
            meta = result.metadata
            
            # Format file info
            file_info = f"{i}. **{meta.filename}**\n"
            file_info += f"   Path: `{meta.full_path}`\n"
            file_info += f"   Type: {meta.file_type} ({meta.extension})\n"
            file_info += f"   Size: {self._format_size(meta.size_bytes)}\n"
            file_info += f"   Modified: {meta.modified_date.strftime('%Y-%m-%d')}\n"
            
            # Add snippets if available
            if result.snippets:
                file_info += "   Content preview:\n"
                for snippet in result.snippets[:2]:
                    # Clean up snippet for display
                    clean_snippet = snippet.replace("\n", " ").strip()
                    if len(clean_snippet) > 200:
                        clean_snippet = clean_snippet[:200] + "..."
                    file_info += f"   > {clean_snippet}\n"
            
            response_parts.append(file_info)
        
        # Add summary
        if intent.notes:
            response_parts.append(f"\n*Search notes: {intent.notes}*")
        
        return "\n".join(response_parts)
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable form."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
