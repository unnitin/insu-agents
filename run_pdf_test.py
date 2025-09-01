#!/usr/bin/env python3
"""
Runner script for the PDF + Orchestrator integration test.

This script sets up the environment and runs the comprehensive integration test
that demonstrates reading an insurance policy PDF and processing it through
the orchestrator pipeline.
"""

import os
import sys
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are available."""
    missing = []
    
    try:
        import PyPDF2
    except ImportError:
        missing.append("PyPDF2")
    
    # Optional but recommended
    try:
        import reportlab
    except ImportError:
        print("⚠️  reportlab not found - will create text file instead of PDF")
    
    try:
        import pdfplumber
    except ImportError:
        print("⚠️  pdfplumber not found - PDF processing may be limited")
    
    if missing:
        print(f"❌ Missing required dependencies: {', '.join(missing)}")
        print("Install with: pip install PyPDF2 reportlab pdfplumber")
        return False
    
    return True

def main():
    """Run the PDF integration test."""
    print("🚀 ORCHESTRATOR PDF INTEGRATION TEST RUNNER")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("Please install missing dependencies and try again.")
        sys.exit(1)
    
    # Set up environment
    project_root = Path(__file__).parent
    src_dir = project_root / "src"
    
    # Add src to Python path
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    # Set PYTHONPATH for subprocess calls
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if str(src_dir) not in current_pythonpath:
        os.environ['PYTHONPATH'] = f"{src_dir}:{current_pythonpath}" if current_pythonpath else str(src_dir)
    
    print(f"📁 Project root: {project_root}")
    print(f"📁 Source directory: {src_dir}")
    print(f"🐍 Python path configured")
    
    # Import and run the test
    try:
        from tests.test_orchestrator_pdf_integration import test_orchestrator_pdf_integration
        
        print(f"\n🧪 Starting integration test...")
        success = test_orchestrator_pdf_integration()
        
        if success:
            print(f"\n🎉 Integration test completed successfully!")
            print(f"\nℹ️  Test artifacts created in: {project_root / 'test_data'}")
            print(f"   - test_policy.pdf (sample insurance policy)")
            print(f"   - orchestrator_test_results.json (detailed results)")
        else:
            print(f"\n❌ Integration test failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Error running test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
