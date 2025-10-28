#!/usr/bin/env python3
"""
Insurance Agents Main Application

Main application flow for insurance quote processing:
1. Import files and initialize registry
2. Scan PDFs and files to understand current insurance requirements  
3. Use research agents to put together quote options
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from dataclasses import asdict
from typing import Dict, List, Any, Optional

# Add src to Python path
project_root = Path(__file__).parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

# Import core components
from data_models import WorldState, PolicySummary, Vehicle, Property
from toolregistry import ToolRegistry
from tools.pdf_reader_tool import PdfReaderTool
from tools.card_ocr_tool import CardOCRTool
from tools.web_research_tool import WebResearchTool
from tools.asset_research_tool import AssetResearchTool
from tools.email_draft_tool import EmailDraftTool

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print application banner."""
    print("=" * 60)
    print("INSURANCE QUOTE PROCESSING SYSTEM")
    print("=" * 60)
    print("Automated insurance analysis and quote generation")
    print("=" * 60)


def step1_initialize_system() -> ToolRegistry:
    """
    Step 1: Import files and initialize registry
    
    Returns:
        ToolRegistry: Configured tool registry with all available tools
    """
    logger.info("Step 1: Initializing system and tool registry")
    
    # Create registry with all available tools
    tools = [
        PdfReaderTool(),
        CardOCRTool(),
        WebResearchTool(),
        AssetResearchTool(),
        EmailDraftTool()
    ]
    
    registry = ToolRegistry(tools)
    
    logger.info(f"Initialized registry with {registry.count()} tools")
    for tool_name in registry.list():
        tool = registry.get(tool_name)
        description = getattr(tool, 'description', 'No description')
        logger.info(f"  - {tool_name}: {description}")
    
    return registry


def step2_analyze_current_insurance(registry: ToolRegistry, 
                                    input_dir: str = "input",
                                    pdf_files: List[str] = None,
                                    zip_code: str = None) -> WorldState:
    """
    Step 2: Scan PDFs and files to understand current insurance requirements
    
    Args:
        registry: Tool registry for accessing processing tools
        input_dir: Directory containing input files
        pdf_files: Specific PDF files to process (optional)
        zip_code: User's ZIP code for location-based processing
    
    Returns:
        WorldState: Initial world state with extracted insurance information
    """
    logger.info("Step 2: Analyzing current insurance requirements")
    
    # Initialize world state
    state = WorldState(user_zip=zip_code)
    
    # Process PDF files for policy information
    pdf_tool = registry.get("pdf_reader")
    
    # Find PDF files to process
    pdf_paths = []
    if pdf_files:
        pdf_paths = pdf_files
    else:
        # Scan input directory for PDFs
        input_path = Path(input_dir)
        if input_path.exists():
            pdf_paths.extend(list(input_path.glob("**/*.pdf")))
    
    logger.info(f"Found {len(pdf_paths)} PDF files to process")
    
    # Process each PDF file
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            logger.warning(f"PDF file not found: {pdf_path}")
            continue
            
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            result = pdf_tool.run(pdf_path=str(pdf_path))
            
            # Extract policy information
            policy_data = result.get('policy', {})
            if policy_data:
                # Update world state with policy information
                if policy_data.get('carrier'):
                    state.policy.carrier = policy_data['carrier']
                if policy_data.get('policy_number'):
                    state.policy.policy_number = policy_data['policy_number']
                if policy_data.get('premium'):
                    state.policy.premium = policy_data['premium']
                if policy_data.get('effective_date'):
                    state.policy.effective_date = policy_data['effective_date']
                if policy_data.get('expiration_date'):
                    state.policy.expiration_date = policy_data['expiration_date']
                
                logger.info(f"Extracted policy info: {policy_data.get('carrier', 'Unknown')} - {policy_data.get('policy_number', 'No policy number')}")
            
            # Store raw text for further analysis
            raw_text = result.get('raw_text', '')
            if raw_text:
                state.policy.raw_text_refs.append(str(pdf_path))
                
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
    
    # Process insurance card images if available
    try:
        card_tool = registry.get("card_ocr")
        images_dir = f"{input_dir}/images"
        
        if os.path.exists(images_dir):
            logger.info(f"Processing insurance card images from {images_dir}")
            card_result = card_tool.run(images_dir=images_dir)
            
            cards = card_result.get('cards', [])
            logger.info(f"Processed {len(cards)} insurance cards")
            
            # Extract additional policy information from cards
            for card_data in cards:
                parsed = card_data.get('parsed', {})
                if parsed.get('policy_number') and not state.policy.policy_number:
                    state.policy.policy_number = parsed['policy_number']
                if parsed.get('carrier') and not state.policy.carrier:
                    state.policy.carrier = parsed['carrier']
                    
    except Exception as e:
        logger.warning(f"Insurance card processing failed: {e}")
    
    # Analyze assets using asset research tool
    try:
        asset_tool = registry.get("asset_research")
        
        # Create a prompt from available policy information
        prompt_parts = []
        if state.policy.carrier:
            prompt_parts.append(f"Current insurance carrier: {state.policy.carrier}")
        if state.policy.policy_number:
            prompt_parts.append(f"Policy number: {state.policy.policy_number}")
        
        if prompt_parts:
            prompt = "Analyze insurance needs based on: " + "; ".join(prompt_parts)
            
            logger.info("Analyzing assets and insurance needs")
            asset_result = asset_tool.run(prompt=prompt, max_results=10)
            
            # Extract vehicles
            vehicles_data = asset_result.get('vehicles', [])
            for vehicle_data in vehicles_data:
                vehicle = Vehicle(
                    make=vehicle_data.get('make', ''),
                    model=vehicle_data.get('model', ''),
                    year=vehicle_data.get('year', ''),
                    vin=vehicle_data.get('vin', '')
                )
                state.vehicles.append(vehicle)
            
            # Extract properties
            properties_data = asset_result.get('properties', [])
            for property_data in properties_data:
                property_obj = Property(
                    property_type=property_data.get('type', ''),
                    address=property_data.get('address', ''),
                    year_built=property_data.get('year_built', ''),
                    square_footage=property_data.get('square_footage', '')
                )
                state.properties.append(property_obj)
            
            logger.info(f"Identified {len(state.vehicles)} vehicles and {len(state.properties)} properties")
            
    except Exception as e:
        logger.warning(f"Asset analysis failed: {e}")
    
    # Log current state summary
    summary = state.get_summary()
    logger.info("Current insurance analysis complete:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")
    
    return state


def step3_research_quote_options(registry: ToolRegistry, state: WorldState) -> WorldState:
    """
    Step 3: Use research agents to put together quote options
    
    Args:
        registry: Tool registry for accessing research tools
        state: Current world state with insurance requirements
    
    Returns:
        WorldState: Updated state with quote options and leads
    """
    logger.info("Step 3: Researching quote options")
    
    if not state.user_zip:
        logger.error("No ZIP code provided for location-based research")
        return state
    
    # Research local insurance providers
    try:
        web_tool = registry.get("web_research")
        
        # Determine asset types for research
        asset_types = []
        if state.vehicles:
            asset_types.append("auto")
        if state.properties:
            asset_types.append("home")
        if not asset_types:
            asset_types = ["auto", "home"]  # Default to both
        
        logger.info(f"Researching {asset_types} insurance providers in ZIP {state.user_zip}")
        
        research_result = web_tool.run(
            zip=state.user_zip,
            asset_types=asset_types,
            top_k=10
        )
        
        leads = research_result.get('leads', [])
        logger.info(f"Found {len(leads)} potential insurance providers")
        
        # Add leads to world state
        from data_models import Lead
        for lead_data in leads:
            lead = Lead(
                name=lead_data.get('name', ''),
                url=lead_data.get('url', ''),
                phone=lead_data.get('phone', ''),
                email=lead_data.get('email', ''),
                notes=lead_data.get('notes', '')
            )
            state.leads.append(lead)
            logger.info(f"Added lead: {lead.name}")
            
    except Exception as e:
        logger.error(f"Web research failed: {e}")
    
    # Generate quote request emails for top leads
    try:
        email_tool = registry.get("email_draft")
        
        # Process top 3 leads for quote requests
        top_leads = state.leads[:3]
        
        for lead in top_leads:
            if not lead.name:
                continue
                
            logger.info(f"Generating quote request email for {lead.name}")
            
            try:
                email_result = email_tool.run(
                    recipient_name=lead.name,
                    sender_name="Insurance Customer",  # Could be parameterized
                    zip=state.user_zip,
                    lines=" + ".join(["auto" if state.vehicles else "", "home" if state.properties else ""]).strip(" + "),
                    policy=asdict(state.policy),
                    vehicles=[asdict(v) for v in state.vehicles],
                    properties=[asdict(p) for p in state.properties]
                )
                
                email_content = email_result.get('email_content', '')
                if email_content:
                    # Save email content for later use
                    lead.notes = f"Quote request email generated. Length: {len(email_content)} chars"
                    logger.info(f"Generated email for {lead.name} ({len(email_content)} characters)")
                    
            except Exception as e:
                logger.warning(f"Failed to generate email for {lead.name}: {e}")
    
    except Exception as e:
        logger.error(f"Email generation failed: {e}")
    
    # Log final state
    final_summary = state.get_summary()
    logger.info("Quote research complete:")
    for key, value in final_summary.items():
        logger.info(f"  {key}: {value}")
    
    return state


def save_results(state: WorldState, filename: str = "insurance_analysis_results.json"):
    """
    Save insurance analysis results to file.
    
    Args:
        state: Final world state with all analysis results
        filename: Output filename
    """
    output_file = project_root / "output" / filename
    output_file.parent.mkdir(exist_ok=True)
    
    try:
        # Convert world state to dictionary for JSON serialization
        results = {
            "timestamp": str(Path(__file__).stat().st_mtime),
            "analysis_summary": state.get_summary(),
            "world_state": asdict(state)
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def main():
    """Main application entry point - Insurance Quote Processing System."""
    parser = argparse.ArgumentParser(
        description="Insurance Quote Processing System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Main workflow:
1. Initialize system and import files
2. Scan PDFs and files to understand current insurance requirements
3. Use research agents to put together quote options

Examples:
  python main.py --zip 90210                    # Process insurance for ZIP code
  python main.py --zip 12345 --pdf policy.pdf  # Include specific PDF
  python main.py --input-dir /path/to/files     # Custom input directory
        """
    )
    
    parser.add_argument("--zip", required=True,
                       help="ZIP code for location-based insurance research")
    parser.add_argument("--pdf", nargs="*",
                       help="Specific PDF files to process")
    parser.add_argument("--input-dir", default="input",
                       help="Input directory containing PDFs and images (default: input)")
    parser.add_argument("--save-results", action="store_true", default=True,
                       help="Save analysis results to JSON file")
    parser.add_argument("--output-file", default="insurance_analysis_results.json",
                       help="Output filename for results")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    print_banner()
    
    try:
        # Step 1: Initialize system and tool registry
        registry = step1_initialize_system()
        
        # Step 2: Analyze current insurance requirements
        state = step2_analyze_current_insurance(
            registry=registry,
            input_dir=args.input_dir,
            pdf_files=args.pdf,
            zip_code=args.zip
        )
        
        # Step 3: Research quote options
        final_state = step3_research_quote_options(registry, state)
        
        # Save results
        if args.save_results:
            save_results(final_state, args.output_file)
        
        # Print summary
        print("\n" + "=" * 60)
        print("INSURANCE ANALYSIS COMPLETE")
        print("=" * 60)
        
        summary = final_state.get_summary()
        print(f"ZIP Code: {final_state.user_zip}")
        print(f"Policy Carrier: {final_state.policy.carrier or 'Not identified'}")
        print(f"Vehicles: {summary['vehicles']}")
        print(f"Properties: {summary['properties']}")
        print(f"Insurance Leads: {summary['leads']}")
        print(f"Ready for Quotes: {summary['ready_for_quotes']}")
        
        if summary['leads'] > 0:
            print(f"\nTop insurance providers found:")
            for i, lead in enumerate(final_state.leads[:3], 1):
                contact = lead.get_primary_contact()
                print(f"  {i}. {lead.name} - {contact}")
        
        logger.info("Insurance quote processing completed successfully")
        
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Insurance processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
