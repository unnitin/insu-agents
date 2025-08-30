#!/usr/bin/env python3
"""
Web Researcher Tool

A simple web researcher tool that AI agents can use to search the web, 
extract content from web pages, and gather information. Includes rate limiting,
error handling, and clean text extraction.

Features:
- Web search using DuckDuckGo (no API key required)
- Content extraction from web pages
- Text cleaning and summarization
- Rate limiting to be respectful to websites
- Simple interface for AI agents
"""

import requests
import time
import re
import logging
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, quote_plus
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TODO: add to requirements.txt and remove this try/except block
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    logger.warning("BeautifulSoup not installed. HTML parsing will be limited.")

# TODO: add to requirements.txt and remove this try/except block
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.error("Requests library not installed. Web functionality will be disabled.")


# TODO: Move the dataclass to the data_models folder
@dataclass
class SearchResult:
    """Data class for search results."""
    title: str
    url: str
    snippet: str
    source: str = "web"


@dataclass
class WebContent:
    """Data class for extracted web content."""
    url: str
    title: str
    content: str
    word_count: int
    extracted_at: str
    success: bool = True
    error: Optional[str] = None


class WebResearcher:
    """
    Simple web researcher tool for AI agents.
    
    Provides web search capabilities and content extraction from web pages.
    Includes rate limiting and error handling for responsible web scraping.
    """
    
    def __init__(self, 
                 delay_between_requests: float = 1.0,
                 max_content_length: int = 10000,
                 timeout: int = 10):
        """
        Initialize the web researcher.
        
        Args:
            delay_between_requests (float): Delay between requests in seconds
            max_content_length (int): Maximum content length to extract
            timeout (int): Request timeout in seconds
        """
        if not HAS_REQUESTS:
            raise ImportError("Requests library is required for web functionality")
        
        self.delay_between_requests = delay_between_requests
        self.max_content_length = max_content_length
        self.timeout = timeout
        self.last_request_time = 0
        
        # Common headers to appear more like a browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        logger.info(f"WebResearcher initialized (delay: {delay_between_requests}s, timeout: {timeout}s)")
    
    def _respect_rate_limit(self):
        """Implement rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.delay_between_requests:
            sleep_time = self.delay_between_requests - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def search_duckduckgo(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Search DuckDuckGo for web results.
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results to return
            
        Returns:
            List[SearchResult]: List of search results
        """
        self._respect_rate_limit()
        
        try:
            # Use DuckDuckGo's instant answers API (no authentication required)
            # First get the search token
            search_url = "https://duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_redirect': '1',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            logger.info(f"Searching DuckDuckGo for: {query}")
            
            # For simplicity, we'll use the HTML search and parse results
            search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
            
            response = requests.get(search_url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            results = []
            if HAS_BS4:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Parse DuckDuckGo HTML results
                result_elements = soup.find_all('div', class_='result__body')
                
                for i, element in enumerate(result_elements[:max_results]):
                    try:
                        title_element = element.find('a', class_='result__a')
                        snippet_element = element.find('a', class_='result__snippet')
                        
                        if title_element:
                            title = title_element.get_text(strip=True)
                            url = title_element.get('href', '')
                            snippet = snippet_element.get_text(strip=True) if snippet_element else ""
                            
                            # Clean up DuckDuckGo redirect URLs
                            if url.startswith('/l/?uddg='):
                                # Extract the actual URL from DuckDuckGo's redirect
                                import urllib.parse
                                url = urllib.parse.unquote(url.split('uddg=')[1])
                            
                            results.append(SearchResult(
                                title=title,
                                url=url,
                                snippet=snippet,
                                source="duckduckgo"
                            ))
                    except Exception as e:
                        logger.debug(f"Error parsing search result {i}: {e}")
                        continue
            else:
                # Fallback: extract basic info with regex if BeautifulSoup not available
                logger.warning("BeautifulSoup not available. Using basic regex extraction.")
                # Simple regex to find links (very basic fallback)
                url_pattern = r'<a[^>]*href=["\']([^"\']*)["\'][^>]*>([^<]*)</a>'
                matches = re.findall(url_pattern, response.text)
                
                for i, (url, title) in enumerate(matches[:max_results]):
                    if url.startswith('http'):
                        results.append(SearchResult(
                            title=title.strip(),
                            url=url,
                            snippet="",
                            source="duckduckgo"
                        ))
            
            logger.info(f"Found {len(results)} search results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching DuckDuckGo: {e}")
            return []
    
    def extract_content(self, url: str) -> WebContent:
        """
        Extract clean text content from a web page.
        
        Args:
            url (str): URL to extract content from
            
        Returns:
            WebContent: Extracted content data
        """
        self._respect_rate_limit()
        
        try:
            logger.info(f"Extracting content from: {url}")
            
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            if HAS_BS4:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract title
                title_element = soup.find('title')
                title = title_element.get_text(strip=True) if title_element else "No title"
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    script.decompose()
                
                # Extract main content (try common content containers first)
                content_candidates = [
                    soup.find('main'),
                    soup.find('article'),
                    soup.find('div', class_=re.compile(r'content|main|article|post')),
                    soup.find('div', id=re.compile(r'content|main|article|post')),
                    soup.find('body')
                ]
                
                content_element = None
                for candidate in content_candidates:
                    if candidate:
                        content_element = candidate
                        break
                
                if content_element:
                    # Extract text content
                    content = content_element.get_text(separator=' ', strip=True)
                else:
                    content = soup.get_text(separator=' ', strip=True)
                
            else:
                # Fallback: basic text extraction with regex
                logger.warning("BeautifulSoup not available. Using basic text extraction.")
                
                # Extract title
                title_match = re.search(r'<title[^>]*>([^<]*)</title>', response.text, re.IGNORECASE)
                title = title_match.group(1).strip() if title_match else "No title"
                
                # Remove scripts, styles, and HTML tags
                content = re.sub(r'<script[^>]*>.*?</script>', '', response.text, flags=re.DOTALL | re.IGNORECASE)
                content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
                content = re.sub(r'<[^>]*>', ' ', content)
                content = re.sub(r'\s+', ' ', content).strip()
            
            # Clean up content
            content = self._clean_text(content)
            
            # Truncate if too long
            if len(content) > self.max_content_length:
                content = content[:self.max_content_length] + "..."
                logger.info(f"Content truncated to {self.max_content_length} characters")
            
            word_count = len(content.split())
            
            return WebContent(
                url=url,
                title=title,
                content=content,
                word_count=word_count,
                extracted_at=datetime.now().isoformat(),
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return WebContent(
                url=url,
                title="Error",
                content="",
                word_count=0,
                extracted_at=datetime.now().isoformat(),
                success=False,
                error=str(e)
            )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common navigation text
        text = re.sub(r'\b(Home|About|Contact|Privacy|Terms|Menu|Navigation|Skip to content)\b', '', text, flags=re.IGNORECASE)
        
        # Remove email and phone patterns that might be noise
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Clean up extra spaces again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def research_topic(self, query: str, max_sources: int = 3) -> Dict[str, any]:
        """
        Research a topic by searching and extracting content from multiple sources.
        
        Args:
            query (str): Research query
            max_sources (int): Maximum number of sources to extract content from
            
        Returns:
            Dict: Research results with search results and extracted content
        """
        logger.info(f"Starting research on topic: {query}")
        
        # Search for results
        search_results = self.search_duckduckgo(query, max_results=max_sources * 2)
        
        if not search_results:
            return {
                'query': query,
                'search_results': [],
                'content': [],
                'summary': "No search results found.",
                'researched_at': datetime.now().isoformat()
            }
        
        # Extract content from top results
        extracted_content = []
        successful_extractions = 0
        
        for result in search_results:
            if successful_extractions >= max_sources:
                break
                
            content = self.extract_content(result.url)
            if content.success and content.word_count > 50:  # Only include substantial content
                extracted_content.append(content)
                successful_extractions += 1
            else:
                logger.debug(f"Skipping {result.url}: extraction failed or content too short")
        
        # Create a summary
        total_words = sum(content.word_count for content in extracted_content)
        summary = f"Research completed. Found {len(search_results)} search results, successfully extracted content from {len(extracted_content)} sources ({total_words} total words)."
        
        return {
            'query': query,
            'search_results': [result.__dict__ for result in search_results],
            'content': [content.__dict__ for content in extracted_content],
            'summary': summary,
            'researched_at': datetime.now().isoformat()
        }
    
    def get_page_summary(self, url: str, max_sentences: int = 5) -> str:
        """
        Get a brief summary of a web page.
        
        Args:
            url (str): URL to summarize
            max_sentences (int): Maximum number of sentences in summary
            
        Returns:
            str: Brief summary of the page
        """
        content = self.extract_content(url)
        
        if not content.success:
            return f"Could not extract content from {url}: {content.error}"
        
        # Simple sentence extraction (take first few sentences)
        sentences = re.split(r'[.!?]+', content.content)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
        
        summary_sentences = sentences[:max_sentences]
        summary = '. '.join(summary_sentences)
        
        if len(sentences) > max_sentences:
            summary += "..."
        
        return f"Summary of {content.title}: {summary}"


def main():
    """Example usage of the WebResearcher tool."""
    researcher = WebResearcher()
    
    # Example 1: Simple search
    print("=== Example 1: Search Results ===")
    results = researcher.search_duckduckgo("artificial intelligence news 2024", max_results=3)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.title}")
        print(f"   URL: {result.url}")
        print(f"   Snippet: {result.snippet[:100]}...")
        print()
    
    # Example 2: Extract content from a page
    if results:
        print("=== Example 2: Content Extraction ===")
        content = researcher.extract_content(results[0].url)
        print(f"Title: {content.title}")
        print(f"Word Count: {content.word_count}")
        print(f"Content Preview: {content.content[:300]}...")
        print()
    
    # Example 3: Research a topic
    print("=== Example 3: Topic Research ===")
    research_results = researcher.research_topic("machine learning applications in healthcare", max_sources=2)
    print(f"Query: {research_results['query']}")
    print(f"Summary: {research_results['summary']}")
    print(f"Sources found: {len(research_results['content'])}")
    print()


if __name__ == "__main__":
    main()
