#!/usr/bin/env python3
"""
Web Researcher Demo

Demonstrates how AI agents can use the WebResearcher tool to gather
information from the web. Shows practical examples of searching,
content extraction, and topic research.
"""

import sys
from pathlib import Path

# Add tools directory to path
sys.path.append(str(Path(__file__).parent))

from web_researcher import WebResearcher, SearchResult, WebContent
import json


class AIAgentExample:
    """
    Example AI agent that uses the WebResearcher tool to gather information.
    
    Demonstrates different ways an AI agent can leverage web research capabilities.
    """
    
    def __init__(self):
        """Initialize the AI agent with web research capabilities."""
        self.web_researcher = WebResearcher(
            delay_between_requests=1.5,  # Be respectful to websites
            max_content_length=5000,     # Limit content length for processing
            timeout=15                   # Reasonable timeout
        )
        print("AI Agent initialized with web research capabilities")
    
    def research_insurance_topic(self, topic: str) -> dict:
        """
        Research an insurance-related topic using web search.
        
        Args:
            topic (str): Insurance topic to research
            
        Returns:
            dict: Research results with key findings
        """
        print(f"\n🔍 Researching insurance topic: {topic}")
        
        # Search for relevant information
        search_query = f"{topic} insurance 2024"
        results = self.web_researcher.research_topic(search_query, max_sources=3)
        
        # Process and analyze the results
        key_findings = []
        sources = []
        
        for content in results['content']:
            # Extract key sentences (simple approach)
            sentences = content['content'].split('. ')
            relevant_sentences = [s for s in sentences if len(s) > 50 and any(
                keyword in s.lower() for keyword in ['insurance', 'coverage', 'policy', 'premium', 'claim']
            )]
            
            if relevant_sentences:
                key_findings.extend(relevant_sentences[:2])  # Top 2 relevant sentences
                sources.append({
                    'title': content['title'],
                    'url': content['url'],
                    'word_count': content['word_count']
                })
        
        return {
            'topic': topic,
            'search_query': search_query,
            'key_findings': key_findings[:5],  # Top 5 findings
            'sources': sources,
            'total_sources': len(results['content']),
            'research_summary': results['summary']
        }
    
    def fact_check_claim(self, claim: str) -> dict:
        """
        Fact-check a claim by searching for supporting information.
        
        Args:
            claim (str): Claim to fact-check
            
        Returns:
            dict: Fact-checking results
        """
        print(f"\n✅ Fact-checking claim: {claim}")
        
        # Search for information about the claim
        search_results = self.web_researcher.search_duckduckgo(claim, max_results=5)
        
        supporting_evidence = []
        contradicting_evidence = []
        
        for result in search_results[:3]:  # Check top 3 results
            content = self.web_researcher.extract_content(result.url)
            
            if content.success:
                # Simple analysis (in a real AI agent, this would be more sophisticated)
                text = content.content.lower()
                claim_keywords = claim.lower().split()
                
                # Look for supporting or contradicting language
                if any(word in text for word in ['true', 'correct', 'accurate', 'confirmed']):
                    supporting_evidence.append({
                        'source': content.title,
                        'url': content.url,
                        'evidence_type': 'supporting'
                    })
                elif any(word in text for word in ['false', 'incorrect', 'myth', 'debunked']):
                    contradicting_evidence.append({
                        'source': content.title,
                        'url': content.url,
                        'evidence_type': 'contradicting'
                    })
        
        return {
            'claim': claim,
            'supporting_evidence': supporting_evidence,
            'contradicting_evidence': contradicting_evidence,
            'sources_checked': len(search_results),
            'confidence': 'low'  # In a real system, this would be calculated
        }
    
    def get_current_trends(self, industry: str = "insurance") -> dict:
        """
        Research current trends in a specific industry.
        
        Args:
            industry (str): Industry to research trends for
            
        Returns:
            dict: Current trends information
        """
        print(f"\n📈 Researching current trends in: {industry}")
        
        # Search for recent trends
        search_query = f"{industry} trends 2024 latest news"
        results = self.web_researcher.research_topic(search_query, max_sources=4)
        
        trends = []
        recent_articles = []
        
        for content in results['content']:
            # Look for trend indicators
            text = content['content']
            trend_keywords = ['trend', 'emerging', 'new', 'growth', 'increase', 'decline', 'shift', 'change']
            
            sentences = text.split('. ')
            trend_sentences = [s for s in sentences if any(keyword in s.lower() for keyword in trend_keywords)]
            
            trends.extend(trend_sentences[:2])  # Top 2 trend sentences per source
            recent_articles.append({
                'title': content['title'],
                'url': content['url'],
                'word_count': content['word_count']
            })
        
        return {
            'industry': industry,
            'search_query': search_query,
            'identified_trends': trends[:6],  # Top 6 trends
            'recent_articles': recent_articles,
            'research_date': results['researched_at']
        }
    
    def competitive_analysis(self, company_name: str) -> dict:
        """
        Perform basic competitive analysis on a company.
        
        Args:
            company_name (str): Company name to analyze
            
        Returns:
            dict: Competitive analysis results
        """
        print(f"\n🏢 Performing competitive analysis for: {company_name}")
        
        # Search for company information
        company_info = self.web_researcher.research_topic(f"{company_name} company overview", max_sources=2)
        
        # Search for competitors
        competitor_info = self.web_researcher.search_duckduckgo(f"{company_name} competitors", max_results=5)
        
        # Search for recent news
        news_results = self.web_researcher.search_duckduckgo(f"{company_name} news 2024", max_results=3)
        
        return {
            'company': company_name,
            'company_overview': company_info['summary'],
            'potential_competitors': [result.title for result in competitor_info],
            'recent_news': [{'title': result.title, 'snippet': result.snippet} for result in news_results],
            'analysis_date': company_info['researched_at']
        }


def demo_ai_agent_capabilities():
    """Demonstrate the AI agent's web research capabilities."""
    print("=" * 60)
    print("🤖 AI Agent Web Research Demo")
    print("=" * 60)
    
    # Initialize the AI agent
    agent = AIAgentExample()
    
    # Example 1: Research an insurance topic
    insurance_research = agent.research_insurance_topic("cyber insurance for small businesses")
    print(f"\nInsurance Research Results:")
    print(f"Key Findings: {len(insurance_research['key_findings'])}")
    for i, finding in enumerate(insurance_research['key_findings'][:3], 1):
        print(f"  {i}. {finding[:100]}...")
    
    # Example 2: Fact-check a claim
    fact_check = agent.fact_check_claim("Most home insurance policies cover flood damage")
    print(f"\nFact-Check Results:")
    print(f"Supporting evidence: {len(fact_check['supporting_evidence'])}")
    print(f"Contradicting evidence: {len(fact_check['contradicting_evidence'])}")
    
    # Example 3: Get current trends
    trends = agent.get_current_trends("insurance technology")
    print(f"\nTrend Analysis:")
    print(f"Identified trends: {len(trends['identified_trends'])}")
    for i, trend in enumerate(trends['identified_trends'][:2], 1):
        print(f"  {i}. {trend[:80]}...")
    
    # Example 4: Competitive analysis
    competitive_analysis = agent.competitive_analysis("State Farm")
    print(f"\nCompetitive Analysis:")
    print(f"Company: {competitive_analysis['company']}")
    print(f"Recent news items: {len(competitive_analysis['recent_news'])}")
    
    print("\n" + "=" * 60)
    print("✅ Demo completed successfully!")
    print("=" * 60)


def save_demo_results():
    """Save demo results to a file for later analysis."""
    agent = AIAgentExample()
    
    # Collect research data
    demo_data = {
        'insurance_research': agent.research_insurance_topic("artificial intelligence in insurance"),
        'fact_check': agent.fact_check_claim("Insurance companies use AI for fraud detection"),
        'trends': agent.get_current_trends("insurtech"),
        'competitive_analysis': agent.competitive_analysis("Progressive Insurance")
    }
    
    # Save to file
    output_file = Path(__file__).parent.parent / 'output' / 'web_research_demo_results.json'
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(demo_data, f, indent=2)
    
    print(f"Demo results saved to: {output_file}")
    return str(output_file)


if __name__ == "__main__":
    print("Choose demo mode:")
    print("1. Interactive demo")
    print("2. Save results to file")
    print("3. Quick test")
    
    choice = input("Enter choice (1-3, or press Enter for quick test): ").strip()
    
    if choice == "1":
        demo_ai_agent_capabilities()
    elif choice == "2":
        save_demo_results()
    else:
        # Quick test
        print("Running quick test...")
        researcher = WebResearcher()
        results = researcher.search_duckduckgo("insurance news", max_results=2)
        print(f"Found {len(results)} search results")
        if results:
            print(f"First result: {results[0].title}")
        print("Quick test completed ✅")
