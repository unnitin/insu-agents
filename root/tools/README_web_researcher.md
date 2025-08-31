# Web Researcher Tool

A simple web researcher tool that AI agents can use to search the web, extract content from web pages, and gather information. Designed to be respectful to websites with built-in rate limiting and error handling.

## Features

- **Web Search**: Uses DuckDuckGo search (no API key required)
- **Content Extraction**: Clean text extraction from web pages
- **Rate Limiting**: Respectful delays between requests
- **Error Handling**: Graceful failure handling
- **AI Agent Friendly**: Simple interface designed for AI agents

## Quick Start

```python
from web_researcher import WebResearcher

# Initialize the researcher
researcher = WebResearcher()

# Search the web
results = researcher.search_duckduckgo("artificial intelligence in insurance", max_results=5)

# Extract content from a webpage
content = researcher.extract_content("https://example.com")

# Research a topic comprehensively
research = researcher.research_topic("cyber insurance trends", max_sources=3)
```

## API Reference

### WebResearcher Class

#### Constructor
```python
WebResearcher(
    delay_between_requests=1.0,  # Delay between requests (seconds)
    max_content_length=10000,    # Maximum content length to extract
    timeout=10                   # Request timeout (seconds)
)
```

#### Methods

**`search_duckduckgo(query, max_results=5)`**
- Search DuckDuckGo for web results
- Returns: `List[SearchResult]`

**`extract_content(url)`**
- Extract clean text content from a web page
- Returns: `WebContent` object

**`research_topic(query, max_sources=3)`**
- Research a topic by searching and extracting content from multiple sources
- Returns: Dictionary with search results and extracted content

**`get_page_summary(url, max_sentences=5)`**
- Get a brief summary of a web page
- Returns: String summary

## Data Classes

### SearchResult
```python
@dataclass
class SearchResult:
    title: str      # Page title
    url: str        # Page URL
    snippet: str    # Search result snippet
    source: str     # Source (e.g., "duckduckgo")
```

### WebContent
```python
@dataclass
class WebContent:
    url: str            # Page URL
    title: str          # Page title
    content: str        # Extracted text content
    word_count: int     # Number of words
    extracted_at: str   # ISO timestamp
    success: bool       # Whether extraction succeeded
    error: str          # Error message (if any)
```

## Example AI Agent Usage

```python
class InsuranceAI:
    def __init__(self):
        self.researcher = WebResearcher(delay_between_requests=1.5)
    
    def research_coverage_type(self, coverage_type):
        """Research a specific type of insurance coverage."""
        query = f"{coverage_type} insurance coverage 2024"
        results = self.researcher.research_topic(query, max_sources=3)
        
        # Process results for AI analysis
        key_points = []
        for content in results['content']:
            # Extract relevant information
            sentences = content['content'].split('. ')
            relevant = [s for s in sentences if 'coverage' in s.lower()]
            key_points.extend(relevant[:2])
        
        return {
            'coverage_type': coverage_type,
            'key_points': key_points,
            'sources': len(results['content'])
        }
```

## Dependencies

The tool requires these Python packages:
- `requests` - For HTTP requests
- `beautifulsoup4` - For HTML parsing (optional but recommended)

Install with:
```bash
pip install requests beautifulsoup4
```

## Rate Limiting & Ethics

The tool includes built-in rate limiting to be respectful to websites:
- Default 1-second delay between requests
- Configurable timeout and content limits
- User-agent headers to identify as a browser
- Graceful error handling

## Advanced Usage

### Custom Configuration
```python
researcher = WebResearcher(
    delay_between_requests=2.0,    # Slower for sensitive sites
    max_content_length=5000,       # Shorter content for faster processing
    timeout=15                     # Longer timeout for slow sites
)
```

### Error Handling
```python
content = researcher.extract_content(url)
if content.success:
    print(f"Extracted {content.word_count} words")
    print(content.content[:200] + "...")
else:
    print(f"Error: {content.error}")
```

### Topic Research with Analysis
```python
def analyze_insurance_trends():
    researcher = WebResearcher()
    
    # Research current trends
    results = researcher.research_topic("insurance industry trends 2024")
    
    # Analyze the findings
    trend_keywords = ['growth', 'decline', 'emerging', 'technology']
    trends = []
    
    for content in results['content']:
        sentences = content['content'].split('. ')
        trend_sentences = [s for s in sentences 
                          if any(keyword in s.lower() for keyword in trend_keywords)]
        trends.extend(trend_sentences)
    
    return trends[:10]  # Top 10 trend mentions
```

## Demo

Run the demo to see the tool in action:
```bash
python web_researcher_demo.py
```

The demo shows practical examples of how AI agents can use the web researcher for:
- Insurance topic research
- Fact-checking claims
- Trend analysis
- Competitive research

## Limitations

- Uses DuckDuckGo search (may have different results than Google)
- Basic content extraction (doesn't handle JavaScript-heavy sites)
- No authentication for protected content
- English-language optimized

## Contributing

This tool is designed to be simple and extensible. Potential improvements:
- Support for other search engines
- Better content extraction algorithms
- Caching mechanisms
- Advanced text analysis
- Multi-language support
