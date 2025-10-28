#!/usr/bin/env python3
"""
Insurance Agents Main Application Runner

This script demonstrates the complete functionality of the insurance agentic suite,
including PDF processing, tool registry, orchestration, and various insurance tools.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import asdict

# Add src to Python path
project_root = Path(__file__).parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

# Import core components
from data_models import WorldState, PolicySummary
from toolregistry import ToolRegistry
from tools.pdf_reader_tool import PdfReaderTool
from tools.card_ocr_tool import CardOCRTool
from tools.web_research_tool import WebResearchTool
from tools.asset_research_tool import AssetResearchTool
from tools.email_draft_tool import EmailDraftTool
from insurance_orchestrator.orchestrator import run_pipeline


def print_banner():
    """Print application banner."""
    print("=" * 70)
    print("🏠 INSURANCE AGENTIC SUITE - MAIN RUNNER")
    print("=" * 70)
    print("Comprehensive insurance processing with AI-powered tools")
    print("Components: PDF Reader | OCR | Web Research | Orchestrator")
    print("=" * 70)


def demo_tool_registry():
    """Demonstrate tool registry functionality."""
    print("\n📋 TOOL REGISTRY DEMO")
    print("-" * 40)
    
    # Create registry with all tools
    tools = [
        PdfReaderTool(),
        CardOCRTool(),
        WebResearchTool(),
        AssetResearchTool(),
        EmailDraftTool()
    ]
    
    registry = ToolRegistry(tools)
    
    print(f"✅ Created registry with {registry.count()} tools:")
    for tool_name in registry.list():
        tool = registry.get(tool_name)
        description = getattr(tool, 'description', 'No description')
        print(f"   • {tool_name}: {description}")
    
    return registry


def demo_pdf_processing(pdf_path: str = None):
    """Demonstrate PDF processing capabilities."""
    print("\n📄 PDF PROCESSING DEMO")
    print("-" * 40)
    
    # Use test PDF if available, otherwise create sample
    if not pdf_path:
        test_pdf = project_root / "test_data" / "test_policy.pdf"
        if test_pdf.exists():
            pdf_path = str(test_pdf)
        else:
            print("⚠️  No PDF file specified and no test PDF found")
            print("   To test PDF processing, provide --pdf path/to/policy.pdf")
            return None
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return None
    
    print(f"📖 Processing PDF: {pdf_path}")
    
    try:
        pdf_tool = PdfReaderTool()
        result = pdf_tool.run(pdf_path=pdf_path)
        
        print(f"✅ PDF processed successfully")
        print(f"   Text length: {len(result.get('raw_text', ''))} characters")
        
        policy = result.get('policy', {})
        if policy:
            print("📋 Extracted policy information:")
            for key, value in policy.items():
                if value:
                    print(f"   • {key}: {value}")
        else:
            print("   No structured policy data extracted")
            
        return result
        
    except Exception as e:
        print(f"❌ PDF processing failed: {e}")
        return None


def demo_orchestrator(zip_code: str = "12345", max_iterations: int = 2):
    """Demonstrate orchestrator pipeline."""
    print(f"\n🎯 ORCHESTRATOR DEMO")
    print("-" * 40)
    
    print(f"🚀 Starting orchestrator pipeline for ZIP: {zip_code}")
    
    # Initialize world state
    state = WorldState(user_zip=zip_code)
    
    # Add some sample data to make the demo more interesting
    state.policy = PolicySummary(
        carrier="Sample Insurance Co",
        policy_number="POL-123456",
        premium="$1,200.00"
    )
    
    print("📊 Initial state:")
    print(f"   • ZIP Code: {state.user_zip}")
    print(f"   • Policy: {state.policy.carrier} ({state.policy.policy_number})")
    print(f"   • Assets: {state.get_total_assets()} items")
    
    try:
        # Run orchestrator pipeline
        final_state = run_pipeline(state, max_iters=max_iterations)
        
        print(f"\n✅ Orchestrator completed after {max_iterations} iterations")
        print("📈 Final state summary:")
        summary = final_state.get_summary()
        for key, value in summary.items():
            print(f"   • {key}: {value}")
        
        return final_state
        
    except Exception as e:
        print(f"❌ Orchestrator pipeline failed: {e}")
        return state


def demo_individual_tools():
    """Demonstrate individual tool capabilities."""
    print(f"\n🔧 INDIVIDUAL TOOLS DEMO")
    print("-" * 40)
    
    # Web Research Tool Demo
    print("🌐 Web Research Tool:")
    try:
        web_tool = WebResearchTool()
        # Note: This might not work without proper web access
        print("   • Tool initialized successfully")
        print("   • Ready to search for insurance providers by ZIP code")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Asset Research Tool Demo
    print("\n🏠 Asset Research Tool:")
    try:
        asset_tool = AssetResearchTool()
        print("   • Tool initialized successfully")
        print("   • Ready to research vehicle and property details")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Email Draft Tool Demo
    print("\n✉️ Email Draft Tool:")
    try:
        email_tool = EmailDraftTool()
        sample_result = email_tool.run(
            recipient_name="Insurance Agent",
            sender_name="Demo User",
            zip="12345",
            lines="auto + home",
            policy={"carrier": "Current Insurer"},
            vehicles=[{"make": "Toyota", "model": "Camry", "year": "2020"}],
            properties=[{"type": "Single Family", "address": "123 Main St"}]
        )
        print("   ✅ Sample email generated successfully")
        email_content = sample_result.get('email_content', '')
        if email_content:
            # Show first few lines
            lines = email_content.split('\n')[:5]
            print("   📝 Email preview:")
            for line in lines:
                print(f"      {line}")
            if len(email_content.split('\n')) > 5:
                print("      ...")
    except Exception as e:
        print(f"   ❌ Error: {e}")


def save_results(data: dict, filename: str = "main_demo_results.json"):
    """Save demo results to file."""
    output_file = project_root / "output" / filename
    output_file.parent.mkdir(exist_ok=True)
    
    try:
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {output_file}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Insurance Agentic Suite Main Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run full demo
  python main.py --zip 90210             # Demo with specific ZIP
  python main.py --pdf policy.pdf        # Include PDF processing
  python main.py --tools-only            # Just show individual tools
  python main.py --orchestrator-only     # Just run orchestrator
        """
    )
    
    parser.add_argument("--zip", default="12345", 
                       help="ZIP code for location-based demos")
    parser.add_argument("--pdf", 
                       help="Path to PDF file for processing demo")
    parser.add_argument("--iterations", type=int, default=2,
                       help="Max iterations for orchestrator pipeline")
    parser.add_argument("--tools-only", action="store_true",
                       help="Only demonstrate individual tools")
    parser.add_argument("--orchestrator-only", action="store_true",
                       help="Only run orchestrator pipeline")
    parser.add_argument("--save-results", action="store_true",
                       help="Save demo results to JSON file")
    
    args = parser.parse_args()
    
    print_banner()
    
    results = {
        "demo_type": "full_suite",
        "zip_code": args.zip,
        "timestamp": str(Path(__file__).stat().st_mtime),
        "components": {}
    }
    
    try:
        if args.tools_only:
            results["demo_type"] = "tools_only"
            demo_individual_tools()
            
        elif args.orchestrator_only:
            results["demo_type"] = "orchestrator_only"
            registry = demo_tool_registry()
            final_state = demo_orchestrator(args.zip, args.iterations)
            if final_state:
                results["components"]["orchestrator"] = asdict(final_state)
                
        else:
            # Full demo
            # 1. Tool Registry
            registry = demo_tool_registry()
            results["components"]["registry"] = {
                "tool_count": registry.count(),
                "tools": registry.list()
            }
            
            # 2. PDF Processing (if specified)
            if args.pdf:
                pdf_result = demo_pdf_processing(args.pdf)
                if pdf_result:
                    results["components"]["pdf_processing"] = {
                        "success": True,
                        "text_length": len(pdf_result.get('raw_text', '')),
                        "policy_extracted": bool(pdf_result.get('policy'))
                    }
            
            # 3. Individual Tools
            demo_individual_tools()
            
            # 4. Orchestrator Pipeline
            final_state = demo_orchestrator(args.zip, args.iterations)
            if final_state:
                results["components"]["orchestrator"] = asdict(final_state)
        
        print(f"\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print(f"   Components tested: {len(results['components'])}")
        
        if args.save_results:
            save_results(results)
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\n💡 Next steps:")
    print(f"   • Run tests: python run_tests.py")
    print(f"   • Process your PDF: python main.py --pdf your_policy.pdf")
    print(f"   • Explore tools in src/tools/")
    print(f"   • Check documentation in README files")


if __name__ == "__main__":
    main()
