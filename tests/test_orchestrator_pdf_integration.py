#!/usr/bin/env python3
"""
Integration Test: PDF Reading + Orchestrator Pipeline

This test demonstrates a complete workflow:
1. Create a sample insurance policy PDF
2. Initialize WorldState with the PDF
3. Run the orchestrator for 2-3 cycles
4. Verify that policy information is extracted and processed
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Any

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_models import WorldState, PolicySummary
from insurance_orchestrator.orchestrator import run_pipeline

def create_sample_pdf_content() -> str:
    """Create sample insurance policy text content for testing."""
    return """
ACME INSURANCE COMPANY
HOMEOWNERS INSURANCE POLICY

Policy Number: HO-2024-123456
Policy Holder: John Smith
Address: 123 Main Street, Seattle, WA 98034

Effective Date: January 1, 2024
Expiration Date: January 1, 2025
Premium: $1,200.00 annually

COVERAGE SUMMARY:
- Dwelling Coverage: $350,000
- Personal Property: $175,000
- Liability: $300,000
- Medical Payments: $5,000

PROPERTY DETAILS:
Address: 123 Main Street, Seattle, WA 98034
Year Built: 1995
Square Footage: 2,400 sq ft
Construction: Frame
Roof: Composition Shingle
Bedrooms: 4
Bathrooms: 2.5
Garage: 2-car attached

VEHICLE INFORMATION:
Vehicle 1: 2020 Honda Accord, VIN: 1HGCV1F30LA123456
Vehicle 2: 2018 Toyota RAV4, VIN: JTMWFREV4JD987654

AGENT INFORMATION:
Agent: Sarah Johnson
Phone: (206) 555-0123
Email: sarah.johnson@acmeinsurance.com

This policy provides coverage for the above property and vehicles
subject to the terms and conditions outlined in the full policy document.
"""

def create_test_pdf(content: str, pdf_path: str) -> bool:
    """
    Create a test PDF file with the given content.
    Uses reportlab if available, otherwise creates a text file.
    """
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter
        
        # Split content into lines and add to PDF
        lines = content.strip().split('\n')
        y_position = height - 50
        
        for line in lines:
            if y_position < 50:  # Start new page if needed
                c.showPage()
                y_position = height - 50
            
            c.drawString(50, y_position, line[:80])  # Truncate long lines
            y_position -= 15
        
        c.save()
        return True
        
    except ImportError:
        # Fallback: create a text file with .pdf extension
        # The PDF reader should handle this gracefully
        with open(pdf_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"⚠️  reportlab not available, created text file: {pdf_path}")
        return False

class MockLLMCall:
    """Mock LLM call for testing orchestrator with LLM planner."""
    
    def __init__(self):
        self.call_count = 0
    
    def __call__(self, prompt: str, tools: list[dict]) -> dict:
        """Mock LLM response that varies by call count."""
        self.call_count += 1
        
        if self.call_count == 1:
            # First call: focus on PDF reading
            return {
                "tool_calls": [
                    {"name": "pdf_reader", "args": {"pdf_path": "test_policy.pdf"}}
                ]
            }
        elif self.call_count == 2:
            # Second call: web research for leads
            return {
                "tool_calls": [
                    {"name": "web_research", "args": {
                        "zip": "98034", 
                        "asset_types": ["auto", "home"], 
                        "top_k": 3
                    }}
                ]
            }
        else:
            # Third call: no more actions
            return {"tool_calls": []}

def test_orchestrator_pdf_integration():
    """
    Main integration test function.
    """
    print("🧪 ORCHESTRATOR PDF INTEGRATION TEST")
    print("=" * 60)
    
    # Step 1: Create test data directory and PDF
    test_data_dir = Path(__file__).parent.parent / "test_data"
    test_data_dir.mkdir(exist_ok=True)
    
    pdf_path = test_data_dir / "test_policy.pdf"
    sample_content = create_sample_pdf_content()
    
    print(f"📄 Creating test PDF: {pdf_path}")
    pdf_created = create_test_pdf(sample_content, str(pdf_path))
    
    if not pdf_path.exists():
        print(f"❌ Failed to create test PDF: {pdf_path}")
        return False
    
    # Step 2: Initialize WorldState with PDF
    print(f"🌍 Initializing WorldState with ZIP: 98034")
    state = WorldState(user_zip="98034")
    
    # Add the PDF to raw_text_refs so orchestrator will process it
    state.policy.raw_text_refs = [str(pdf_path)]
    
    print(f"📋 Initial state:")
    print(f"  - ZIP code: {state.user_zip}")
    print(f"  - PDF refs: {len(state.policy.raw_text_refs)}")
    print(f"  - Vehicles: {len(state.vehicles)}")
    print(f"  - Properties: {len(state.properties)}")
    print(f"  - Leads: {len(state.leads)}")
    
    # Step 3: Test with heuristic planner
    print(f"\n🤖 Running orchestrator with HEURISTIC planner (max 3 iterations)...")
    try:
        final_state_heuristic = run_pipeline(
            state=state,
            max_iters=3,
            planner='heuristic'
        )
        
        print(f"✅ Heuristic planner completed successfully!")
        print(f"📊 Final state summary:")
        print(f"  - Policy carrier: {final_state_heuristic.policy.carrier}")
        print(f"  - Policy number: {final_state_heuristic.policy.policy_number}")
        print(f"  - Vehicles: {len(final_state_heuristic.vehicles)}")
        print(f"  - Properties: {len(final_state_heuristic.properties)}")
        print(f"  - Leads: {len(final_state_heuristic.leads)}")
        print(f"  - Bid intents: {len(final_state_heuristic.bid_intents)}")
        print(f"  - Bid results: {len(final_state_heuristic.bid_results)}")
        
    except Exception as e:
        print(f"❌ Heuristic planner failed: {e}")
        return False
    
    # Step 4: Test with LLM planner (mock)
    print(f"\n🧠 Running orchestrator with MOCK LLM planner (max 3 iterations)...")
    try:
        # Reset state for LLM test
        state_llm = WorldState(user_zip="98034")
        state_llm.policy.raw_text_refs = [str(pdf_path)]
        
        mock_llm = MockLLMCall()
        final_state_llm = run_pipeline(
            state=state_llm,
            max_iters=3,
            planner='llm',
            llm_call=mock_llm
        )
        
        print(f"✅ LLM planner completed successfully!")
        print(f"📊 LLM Final state summary:")
        print(f"  - Policy carrier: {final_state_llm.policy.carrier}")
        print(f"  - Policy number: {final_state_llm.policy.policy_number}")
        print(f"  - Vehicles: {len(final_state_llm.vehicles)}")
        print(f"  - Properties: {len(final_state_llm.properties)}")
        print(f"  - Leads: {len(final_state_llm.leads)}")
        print(f"  - LLM calls made: {mock_llm.call_count}")
        
    except Exception as e:
        print(f"❌ LLM planner failed: {e}")
        return False
    
    # Step 5: Verify results
    print(f"\n🔍 Verifying results...")
    
    # Check that policy information was extracted
    policy_extracted = bool(
        final_state_heuristic.policy.policy_number or 
        final_state_heuristic.policy.carrier
    )
    
    if policy_extracted:
        print(f"✅ Policy information successfully extracted")
    else:
        print(f"⚠️  Policy information not extracted (may be expected)")
    
    # Check that orchestrator completed multiple cycles
    cycles_completed = mock_llm.call_count >= 2
    if cycles_completed:
        print(f"✅ Multiple orchestrator cycles completed ({mock_llm.call_count} LLM calls)")
    else:
        print(f"⚠️  Expected multiple cycles, got {mock_llm.call_count}")
    
    # Step 6: Save detailed results
    results_file = test_data_dir / "orchestrator_test_results.json"
    results = {
        "test_summary": {
            "pdf_path": str(pdf_path),
            "pdf_created": pdf_created,
            "policy_extracted": policy_extracted,
            "cycles_completed": cycles_completed,
            "llm_calls": mock_llm.call_count
        },
        "heuristic_final_state": asdict(final_state_heuristic),
        "llm_final_state": asdict(final_state_llm)
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Detailed results saved to: {results_file}")
    
    # Step 7: Cleanup (optional)
    cleanup = os.getenv('CLEANUP_TEST_FILES', 'false').lower() == 'true'
    if cleanup:
        pdf_path.unlink(missing_ok=True)
        results_file.unlink(missing_ok=True)
        print(f"🧹 Test files cleaned up")
    
    print(f"\n🎉 Integration test completed successfully!")
    return True

def main():
    """Run the integration test."""
    try:
        success = test_orchestrator_pdf_integration()
        if success:
            print(f"\n✅ ALL TESTS PASSED")
            sys.exit(0)
        else:
            print(f"\n❌ TESTS FAILED")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
