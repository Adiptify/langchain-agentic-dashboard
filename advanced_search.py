"""
Advanced search filters and autocomplete system for the LangChain agentic dashboard.
Provides intelligent search suggestions, filtering, and query optimization.
"""

import re
import json
import sqlite3
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import uuid

@dataclass
class SearchFilter:
    """Search filter configuration"""
    filter_id: str
    name: str
    filter_type: str  # 'date_range', 'file_type', 'content_type', 'numeric_range', 'text_match'
    parameters: Dict[str, Any]
    is_active: bool = True

@dataclass
class SearchSuggestion:
    """Search suggestion with metadata"""
    text: str
    type: str  # 'query', 'filter', 'file', 'column'
    confidence: float
    metadata: Dict[str, Any]

class AdvancedSearchEngine:
    """
    Advanced search engine with filters, autocomplete, and query optimization.
    """
    
    def __init__(self, db_path: str = "search_index.db"):
        self.db_path = db_path
        self._init_database()
        
        # Common query patterns for autocomplete
        self.query_patterns = [
            r"what is the total",
            r"show me",
            r"find all",
            r"how many",
            r"list",
            r"sum of",
            r"average",
            r"maximum",
            r"minimum",
            r"count",
            r"filter by",
            r"group by",
            r"sort by"
        ]
        
        # Energy-related keywords for domain-specific suggestions
        self.energy_keywords = [
            "energy consumption", "power usage", "electricity", "kwh", "kilowatt",
            "meter reading", "usage", "consumption", "demand", "load", "feeder",
            "panel", "relay", "transformer", "distribution", "generation"
        ]
    
    def _init_database(self):
        """Initialize search index database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Search index table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_index (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                content_type TEXT NOT NULL,
                file_id TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                last_accessed TEXT,
                access_count INTEGER DEFAULT 0
            )
        """)
        
        # Query patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_patterns (
                id TEXT PRIMARY KEY,
                pattern TEXT NOT NULL,
                category TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                success_rate REAL DEFAULT 1.0,
                last_used TEXT
            )
        """)
        
        # Search suggestions cache
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS suggestion_cache (
                id TEXT PRIMARY KEY,
                partial_query TEXT NOT NULL,
                suggestions TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def index_content(self, content: str, content_type: str, file_id: str, 
                     metadata: Dict[str, Any] = None):
        """Index content for search"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Extract searchable terms
        searchable_content = self._extract_searchable_terms(content)
        
        cursor.execute("""
            INSERT OR REPLACE INTO search_index 
            (id, content, content_type, file_id, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()),
            searchable_content,
            content_type,
            file_id,
            json.dumps(metadata or {}),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _extract_searchable_terms(self, content: str) -> str:
        """Extract searchable terms from content"""
        # Convert to lowercase and extract meaningful terms
        content_lower = content.lower()
        
        # Extract numbers, dates, and keywords
        terms = []
        
        # Extract dates (various formats)
        date_patterns = [
            r'\d{1,2}-\d{1,2}-\d{4}',  # DD-MM-YYYY
            r'\d{4}-\d{1,2}-\d{1,2}',  # YYYY-MM-DD
            r'\d{1,2}/\d{1,2}/\d{4}',  # DD/MM/YYYY
        ]
        
        for pattern in date_patterns:
            dates = re.findall(pattern, content)
            terms.extend(dates)
        
        # Extract numbers
        numbers = re.findall(r'\d+(?:\.\d+)?', content)
        terms.extend(numbers)
        
        # Extract keywords (words with 3+ characters)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content_lower)
        terms.extend(words)
        
        # Extract special terms (like SWB, KWH, etc.)
        special_terms = re.findall(r'\b[A-Z]{2,}\b', content)
        terms.extend(special_terms)
        
        return ' '.join(set(terms))  # Remove duplicates
    
    def get_autocomplete_suggestions(self, partial_query: str, limit: int = 10) -> List[SearchSuggestion]:
        """Get autocomplete suggestions for partial query"""
        suggestions = []
        partial_lower = partial_query.lower().strip()
        
        if len(partial_lower) < 2:
            return suggestions
        
        # Check cache first
        cached_suggestions = self._get_cached_suggestions(partial_lower)
        if cached_suggestions:
            return cached_suggestions
        
        # Generate suggestions from different sources
        suggestions.extend(self._get_query_suggestions(partial_lower, limit))
        suggestions.extend(self._get_content_suggestions(partial_lower, limit))
        suggestions.extend(self._get_filter_suggestions(partial_lower, limit))
        
        # Sort by confidence and remove duplicates
        suggestions = self._deduplicate_suggestions(suggestions)
        suggestions = sorted(suggestions, key=lambda x: x.confidence, reverse=True)[:limit]
        
        # Cache the results
        self._cache_suggestions(partial_lower, suggestions)
        
        return suggestions
    
    def _get_query_suggestions(self, partial_query: str, limit: int) -> List[SearchSuggestion]:
        """Get query pattern suggestions"""
        suggestions = []
        
        for pattern in self.query_patterns:
            if pattern.startswith(partial_query):
                suggestions.append(SearchSuggestion(
                    text=pattern,
                    type="query",
                    confidence=0.9,
                    metadata={"pattern": pattern}
                ))
        
        # Add domain-specific suggestions
        for keyword in self.energy_keywords:
            if keyword.startswith(partial_query):
                suggestions.append(SearchSuggestion(
                    text=f"total {keyword}",
                    type="query",
                    confidence=0.8,
                    metadata={"domain": "energy"}
                ))
        
        return suggestions[:limit//3]
    
    def _get_content_suggestions(self, partial_query: str, limit: int) -> List[SearchSuggestion]:
        """Get suggestions from indexed content"""
        suggestions = []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Search for matching content
        cursor.execute("""
            SELECT content, content_type, file_id, metadata, access_count
            FROM search_index 
            WHERE content LIKE ?
            ORDER BY access_count DESC, created_at DESC
            LIMIT ?
        """, (f"%{partial_query}%", limit//3))
        
        for row in cursor.fetchall():
            content, content_type, file_id, metadata, access_count = row
            
            # Extract relevant terms
            terms = content.split()
            matching_terms = [term for term in terms if partial_query in term.lower()]
            
            for term in matching_terms[:3]:  # Limit to 3 terms per content
                suggestions.append(SearchSuggestion(
                    text=term,
                    type="content",
                    confidence=0.7 + (access_count * 0.01),  # Boost by access count
                    metadata={
                        "content_type": content_type,
                        "file_id": file_id,
                        "original_metadata": json.loads(metadata) if metadata else {}
                    }
                ))
        
        conn.close()
        return suggestions[:limit//3]
    
    def _get_filter_suggestions(self, partial_query: str, limit: int) -> List[SearchSuggestion]:
        """Get filter suggestions"""
        suggestions = []
        
        filter_types = [
            ("date range", "Filter by date range"),
            ("file type", "Filter by file type"),
            ("content type", "Filter by content type"),
            ("numeric range", "Filter by numeric range"),
            ("text match", "Filter by text match")
        ]
        
        for filter_type, description in filter_types:
            if filter_type.startswith(partial_query):
                suggestions.append(SearchSuggestion(
                    text=f"filter by {filter_type}",
                    type="filter",
                    confidence=0.6,
                    metadata={"filter_type": filter_type, "description": description}
                ))
        
        return suggestions[:limit//3]
    
    def _deduplicate_suggestions(self, suggestions: List[SearchSuggestion]) -> List[SearchSuggestion]:
        """Remove duplicate suggestions"""
        seen = set()
        unique_suggestions = []
        
        for suggestion in suggestions:
            if suggestion.text not in seen:
                seen.add(suggestion.text)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions
    
    def _get_cached_suggestions(self, partial_query: str) -> Optional[List[SearchSuggestion]]:
        """Get cached suggestions if available and not expired"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT suggestions, expires_at
            FROM suggestion_cache 
            WHERE partial_query = ? AND expires_at > ?
        """, (partial_query, datetime.now().isoformat()))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            suggestions_data = json.loads(row[0])
            return [
                SearchSuggestion(
                    text=s["text"],
                    type=s["type"],
                    confidence=s["confidence"],
                    metadata=s["metadata"]
                )
                for s in suggestions_data
            ]
        
        return None
    
    def _cache_suggestions(self, partial_query: str, suggestions: List[SearchSuggestion]):
        """Cache suggestions for future use"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        suggestions_data = [
            {
                "text": s.text,
                "type": s.type,
                "confidence": s.confidence,
                "metadata": s.metadata
            }
            for s in suggestions
        ]
        
        expires_at = (datetime.now() + timedelta(hours=1)).isoformat()
        
        cursor.execute("""
            INSERT OR REPLACE INTO suggestion_cache 
            (id, partial_query, suggestions, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()),
            partial_query,
            json.dumps(suggestions_data),
            datetime.now().isoformat(),
            expires_at
        ))
        
        conn.commit()
        conn.close()
    
    def apply_filters(self, query: str, filters: List[SearchFilter]) -> Tuple[str, Dict[str, Any]]:
        """Apply search filters to query"""
        filter_metadata = {}
        modified_query = query
        
        for filter_obj in filters:
            if not filter_obj.is_active:
                continue
            
            if filter_obj.filter_type == "date_range":
                start_date = filter_obj.parameters.get("start_date")
                end_date = filter_obj.parameters.get("end_date")
                if start_date and end_date:
                    modified_query += f" date:{start_date} to {end_date}"
                    filter_metadata["date_range"] = {"start": start_date, "end": end_date}
            
            elif filter_obj.filter_type == "file_type":
                file_types = filter_obj.parameters.get("file_types", [])
                if file_types:
                    file_type_str = " ".join([f"filetype:{ft}" for ft in file_types])
                    modified_query += f" {file_type_str}"
                    filter_metadata["file_types"] = file_types
            
            elif filter_obj.filter_type == "content_type":
                content_types = filter_obj.parameters.get("content_types", [])
                if content_types:
                    content_type_str = " ".join([f"contenttype:{ct}" for ct in content_types])
                    modified_query += f" {content_type_str}"
                    filter_metadata["content_types"] = content_types
            
            elif filter_obj.filter_type == "numeric_range":
                field = filter_obj.parameters.get("field")
                min_val = filter_obj.parameters.get("min_value")
                max_val = filter_obj.parameters.get("max_value")
                if field and min_val is not None and max_val is not None:
                    modified_query += f" {field}:{min_val} to {max_val}"
                    filter_metadata["numeric_range"] = {
                        "field": field,
                        "min": min_val,
                        "max": max_val
                    }
            
            elif filter_obj.filter_type == "text_match":
                field = filter_obj.parameters.get("field")
                value = filter_obj.parameters.get("value")
                if field and value:
                    modified_query += f" {field}:{value}"
                    filter_metadata["text_match"] = {"field": field, "value": value}
        
        return modified_query, filter_metadata
    
    def optimize_query(self, query: str) -> str:
        """Optimize query for better search results"""
        # Convert to lowercase for processing
        query_lower = query.lower()
        
        # Expand abbreviations
        abbreviations = {
            "kwh": "kilowatt hour",
            "swb": "switchboard",
            "i/c": "incoming",
            "o/c": "outgoing",
            "mtr": "meter",
            "pwr": "power",
            "elec": "electricity"
        }
        
        for abbr, full_form in abbreviations.items():
            query_lower = query_lower.replace(abbr, f"{abbr} {full_form}")
        
        # Add synonyms for energy-related terms
        synonyms = {
            "consumption": "consumption usage",
            "usage": "usage consumption",
            "power": "power electricity energy",
            "energy": "energy power electricity"
        }
        
        for term, synonyms_str in synonyms.items():
            if term in query_lower:
                query_lower = query_lower.replace(term, synonyms_str)
        
        return query_lower
    
    def log_query_pattern(self, query: str, success: bool):
        """Log query pattern for learning"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Extract pattern from query
        pattern = self._extract_query_pattern(query)
        
        if pattern:
            cursor.execute("""
                INSERT OR REPLACE INTO query_patterns 
                (id, pattern, category, frequency, success_rate, last_used)
                VALUES (?, ?, ?, 
                    COALESCE((SELECT frequency FROM query_patterns WHERE pattern = ?), 0) + 1,
                    COALESCE((SELECT success_rate FROM query_patterns WHERE pattern = ?), 1.0) * 0.9 + (0.1 if ? else 0),
                    ?)
            """, (
                str(uuid.uuid4()),
                pattern,
                self._categorize_pattern(pattern),
                pattern,
                pattern,
                success,
                datetime.now().isoformat()
            ))
        
        conn.commit()
        conn.close()
    
    def _extract_query_pattern(self, query: str) -> Optional[str]:
        """Extract pattern from query"""
        query_lower = query.lower()
        
        for pattern in self.query_patterns:
            if pattern in query_lower:
                return pattern
        
        return None
    
    def _categorize_pattern(self, pattern: str) -> str:
        """Categorize query pattern"""
        if "total" in pattern or "sum" in pattern:
            return "aggregation"
        elif "show" in pattern or "list" in pattern:
            return "display"
        elif "find" in pattern or "search" in pattern:
            return "search"
        elif "how many" in pattern or "count" in pattern:
            return "counting"
        else:
            return "general"

# Global search engine instance
advanced_search_engine = AdvancedSearchEngine()
