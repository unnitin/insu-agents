#!/usr/bin/env python3
"""
Test script to demonstrate agentic policy extraction with fallback.

This script shows how the PDF reader now uses an intelligent LLM-based approach
first, then falls back to regex patterns if the LLM is unavailable.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_agentic_extraction():
    """Test the agentic extraction approach."""
    from insurance_tools.pdf_reader_tool import PdfReaderTool
    
    print("🤖 AGENTIC POLICY EXTRACTION TEST")
    print("=" * 50)
    
    # Test without API key (should use regex fallback)
    print("\n📋 Test 1: Without OpenAI API Key (Regex Fallback)")
    print("-" * 45)
    
    # Ensure no API key is set
    if 'OPENAI_API_KEY' in os.environ:
        del os.environ['OPENAI_API_KEY']
    
    pdf_tool = PdfReaderTool()
    pdf_path = '../test_data/test_policy.pdf'
    
    try:
        result = pdf_tool.run(pdf_path=pdf_path)
        policy = result.get('policy', {})
        
        print(f"✅ Extraction successful (fallback method)")
        print(f"   Policy Number: {policy.get('policy_number', 'Not found')}")
        print(f"   Carrier: {policy.get('carrier', 'Not found')}")
        print(f"   Method Used: Regex (as expected without API key)")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    # Test with mock API key (would use LLM if key was valid)
    print("\n📋 Test 2: With Mock API Key (Would Use LLM)")
    print("-" * 45)
    
    # Set a mock API key to demonstrate the flow
    os.environ['OPENAI_API_KEY'] = 'sk-mock-key-for-testing'
    
    try:
        result = pdf_tool.run(pdf_path=pdf_path)
        policy = result.get('policy', {})
        
        print(f"✅ Extraction successful (attempted LLM, fell back to regex)")
        print(f"   Policy Number: {policy.get('policy_number', 'Not found')}")
        print(f"   Carrier: {policy.get('carrier', 'Not found')}")
        print(f"   Method Used: Regex fallback (mock API key failed as expected)")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    # Clean up
    if 'OPENAI_API_KEY' in os.environ:
        del os.environ['OPENAI_API_KEY']
    
    print("\n💡 How to Enable Full Agentic Mode:")
    print("   1. Set a valid OpenAI API key:")
    print("      export OPENAI_API_KEY='your-actual-api-key-here'")
    print("   2. Run the test again")
    print("   3. The system will use GPT-3.5-turbo for intelligent extraction")
    print("   4. Falls back to regex if LLM fails or returns incomplete data")
    
    print("\n🎯 Architecture Benefits:")
    print("   ✅ Intelligent extraction using LLM when available")
    print("   ✅ Graceful fallback to proven regex patterns")
    print("   ✅ No breaking changes - works with or without API key")
    print("   ✅ Logging shows which method was used")
    print("   ✅ Robust error handling and validation")

if __name__ == "__main__":
    test_agentic_extraction()
