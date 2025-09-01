#!/usr/bin/env python3
"""
Test Runner for Insurance Orchestrator Project

Simple script to run all tests from the project root.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run all available tests."""
    print("🧪 INSURANCE ORCHESTRATOR TEST RUNNER")
    print("=" * 50)
    
    project_root = Path(__file__).parent
    tests_dir = project_root / "tests"
    
    # Ensure we're in the right directory
    os.chdir(project_root)
    
    # Activate virtual environment if it exists
    venv_activate = project_root / "venv" / "bin" / "activate"
    if venv_activate.exists():
        print("🐍 Using virtual environment")
    else:
        print("⚠️  No virtual environment found - using system Python")
    
    # Available tests
    tests = [
        {
            "name": "Import Test",
            "file": "test_imports.py",
            "description": "Basic import validation"
        },
        {
            "name": "Agentic Extraction Test", 
            "file": "test_agentic_extraction.py",
            "description": "Test LLM-based policy extraction with regex fallback"
        },
        {
            "name": "PDF + Orchestrator Integration Test",
            "file": "run_pdf_test.py", 
            "description": "Full end-to-end test with PDF processing and orchestrator"
        }
    ]
    
    print(f"\n📂 Tests directory: {tests_dir}")
    print(f"📋 Available tests:")
    
    for i, test in enumerate(tests, 1):
        print(f"   {i}. {test['name']}")
        print(f"      {test['description']}")
    
    # Run all tests
    print(f"\n🚀 Running all tests...")
    
    for test in tests:
        test_file = tests_dir / test["file"]
        if not test_file.exists():
            print(f"❌ Test file not found: {test_file}")
            continue
            
        print(f"\n" + "─" * 60)
        print(f"🧪 Running: {test['name']}")
        print(f"📄 File: {test['file']}")
        print("─" * 60)
        
        try:
            # Run the test
            result = subprocess.run([
                sys.executable, str(test_file)
            ], cwd=tests_dir, capture_output=False)
            
            if result.returncode == 0:
                print(f"✅ {test['name']} - PASSED")
            else:
                print(f"❌ {test['name']} - FAILED (exit code: {result.returncode})")
                
        except Exception as e:
            print(f"💥 {test['name']} - ERROR: {e}")
    
    print(f"\n" + "=" * 60)
    print(f"🎉 All tests completed!")
    print(f"\n💡 To run individual tests:")
    print(f"   cd tests")
    print(f"   python test_agentic_extraction.py")
    print(f"   python run_pdf_test.py")

if __name__ == "__main__":
    main()
