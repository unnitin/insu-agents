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
    Represents a single result from a web search operation.
    
    This model stores the essential information returned by search engines
    when looking for insurance-related information, quotes, or providers.
    Used by web research tools to structure and process search results
    before further analysis or content extraction.
    
    The model provides a standardized format for search results regardless
    of the underlying search engine used.
    
    Attributes:
        title: Page title from search results
        url: Full URL of the search result page
        snippet: Brief description/preview text from search results
        source: Search engine source identifier (e.g., "duckduckgo", "google")
    """
    title: str          # Page title from search results
    url: str           # Page URL
    snippet: str       # Search result snippet/description
    source: str = "web"  # Search engine source (e.g., "duckduckgo", "google")


@dataclass
class WebContent:
    """
    Represents content extracted from a web page.
    
    This model stores the text content extracted from web pages, along with
    metadata about the extraction process. Used by web research tools to
    structure and analyze content from insurance provider websites, quote
    forms, and other relevant web resources.
    
    The model includes utility methods for content analysis, keyword searching,
    and text processing to support automated research operations.
    
    Attributes:
        url: Source URL of the extracted content
        title: Page title
        content: Extracted text content from the page
        word_count: Number of words in the extracted content
        extracted_at: ISO timestamp of when extraction occurred
        success: Whether the extraction was successful
        error: Error message if extraction failed
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
