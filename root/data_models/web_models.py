#!/usr/bin/env python3
"""
Web Research Data Models

Contains data classes for web search results and extracted content.
Used by the WebResearcher tool and AI agents for web research operations.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SearchResult:
    """
    Data class for web search results.
    
    Represents a single search result from a web search engine,
    containing the essential information needed for further processing.
    """
    title: str          # Page title from search results
    url: str           # Page URL
    snippet: str       # Search result snippet/description
    source: str = "web"  # Search engine source (e.g., "duckduckgo", "google")


@dataclass
class WebContent:
    """
    Data class for extracted web page content.
    
    Represents the content extracted from a web page, including
    metadata about the extraction process and success status.
    """
    url: str                    # Source URL
    title: str                  # Page title
    content: str               # Extracted text content
    word_count: int            # Number of words in content
    extracted_at: str          # ISO timestamp of extraction
    success: bool = True       # Whether extraction succeeded
    error: Optional[str] = None  # Error message if extraction failed
    
    def get_preview(self, max_length: int = 200) -> str:
        """Get a preview of the content."""
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."
    
    def has_keyword(self, keyword: str) -> bool:
        """Check if content contains a specific keyword (case-insensitive)."""
        return keyword.lower() in self.content.lower()
    
    def get_sentences_with_keyword(self, keyword: str, max_sentences: int = 3) -> list:
        """Get sentences that contain a specific keyword."""
        sentences = self.content.split('. ')
        matching_sentences = [
            s.strip() for s in sentences 
            if keyword.lower() in s.lower() and len(s.strip()) > 20
        ]
        return matching_sentences[:max_sentences]
