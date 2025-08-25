#!/usr/bin/env python3
"""
Prompt-Based Asset Researcher Agent

An AI-powered agent that uses structured prompts to analyze any insurance document 
and build comprehensive fact bases about insured assets. Extracts detailed information
about properties (roof type, house layout, year built, flooring, etc.) and vehicles
(specifications, safety features, usage patterns, etc.).
"""

import re
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict
from datetime import datetime
import logging

# Add config and tools directories to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'config'))
sys.path.append(str(Path(__file__).parent.parent / 'tools'))

# Import asset models from config
from assets import (Vehicle, Property, PersonalItem, PolicySummary, AssetCollection, AnalysisMetadata,
                   VehicleFactBase, PropertyFactBase)

# Import PDF processing capabilities
from pdf_reader import InsurancePDFReader

# Setup logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional AI imports (graceful degradation if not available)
try:
    import openai
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    openai = None
    OpenAI = None
    logger.warning("OpenAI library not installed. OpenAI features will be disabled.")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    AutoTokenizer = None
    AutoModelForCausalLM = None
    pipeline = None
    torch = None
    logger.warning("Transformers library not installed. Open source model features will be disabled.")


class PromptBasedAssetResearcher:
    """
    AI-powered prompt-based agent for analyzing any insurance document and building comprehensive fact bases.
    
    This agent uses structured AI prompts to extract detailed asset information from insurance documents,
    building comprehensive fact bases about properties (roof type, construction details, layout) and 
    vehicles (specifications, safety features, usage patterns). Designed to work with any insurance 
    document type and provide detailed asset intelligence.
    """
    
    def __init__(self, 
                 model_type: str = "auto",
                 model_name: Optional[str] = None,
                 api_key: Optional[str] = None, 
                 input_dir: str = "input/pdf", 
                 output_dir: str = "output",
                 fallback_patterns: bool = True,
                 device: Optional[str] = None):
        """
        Initialize the prompt-based asset researcher agent with support for multiple model backends.
        
        Args:
            model_type (str): Model backend to use ("openai", "huggingface", or "auto")
            model_name (str, optional): Specific model name to use
            api_key (str, optional): OpenAI API key. If not provided, will check environment
            input_dir (str): Directory containing input PDF files
            output_dir (str): Directory for saving processed output
            fallback_patterns (bool): Whether to use pattern matching as fallback
            device (str, optional): Device for HuggingFace models ("cpu", "cuda", "auto")
        """
        self.model_type = model_type
        self.model_name = model_name
        self.device = device or ("cuda" if torch and torch.cuda.is_available() else "cpu")
        self.fallback_patterns = fallback_patterns
        
        # Initialize model backends
        self.openai_client = None
        self.hf_model = None
        self.hf_tokenizer = None
        self.hf_pipeline = None
        self.use_ai = False
        
        # Initialize PDF reader for direct PDF processing
        self.pdf_reader = InsurancePDFReader(input_dir, output_dir)
        
        # Initialize AI backend based on model_type
        if model_type == "auto":
            self._initialize_auto_backend(api_key)
        elif model_type == "openai":
            self._initialize_openai_backend(api_key)
        elif model_type == "huggingface":
            self._initialize_huggingface_backend(model_name)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Use 'openai', 'huggingface', or 'auto'")
        
        # Initialize fallback patterns if requested (always available as backup)
        if self.fallback_patterns:
            self.vehicle_patterns = self._init_vehicle_patterns()
            self.property_patterns = self._init_property_patterns()
            self.policy_patterns = self._init_policy_patterns()
            self.coverage_patterns = self._init_coverage_patterns()
            if not self.use_ai:
                logger.warning("Using fallback pattern matching. Limited functionality available.")
            else:
                logger.debug("Pattern fallback initialized as backup to AI analysis.")
        
        if not self.use_ai and not self.fallback_patterns:
            raise ValueError("Either AI backend must be available or fallback_patterns must be enabled.")
        
        active_backend = "OpenAI" if self.openai_client else "HuggingFace" if self.hf_pipeline else "Patterns"
        logger.info(f"PromptBasedAssetResearcher initialized (Backend: {active_backend}, Device: {self.device})")
    
    def _initialize_auto_backend(self, api_key: Optional[str] = None):
        """Initialize the best available AI backend automatically."""
        # Try OpenAI first (faster for most use cases)
        if HAS_OPENAI:
            try:
                self._initialize_openai_backend(api_key)
                if self.use_ai:
                    logger.info("Auto-selected OpenAI backend")
                    return
            except Exception as e:
                logger.debug(f"OpenAI initialization failed: {e}")
        
        # Fall back to HuggingFace
        if HAS_TRANSFORMERS:
            try:
                self._initialize_huggingface_backend()
                if self.use_ai:
                    logger.info("Auto-selected HuggingFace backend")
                    return
            except Exception as e:
                logger.debug(f"HuggingFace initialization failed: {e}")
        
        logger.warning("No AI backend available. Will use pattern matching fallback.")
    
    def _initialize_openai_backend(self, api_key: Optional[str] = None):
        """Initialize OpenAI backend."""
        if not HAS_OPENAI:
            raise ImportError("OpenAI library not available")
        
        try:
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
            elif os.getenv('OPENAI_API_KEY'):
                self.openai_client = OpenAI()
            else:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
            
            # Test the connection with a simple call
            self.openai_client.models.list()
            self.use_ai = True
            logger.info("OpenAI backend initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI backend: {e}")
            self.openai_client = None
    
    def _initialize_huggingface_backend(self, model_name: Optional[str] = None):
        """Initialize HuggingFace backend with specified or default model."""
        if not HAS_TRANSFORMERS:
            raise ImportError("Transformers library not available")
        
        # Default model recommendations based on task and resources
        if model_name is None:
            model_name = self._get_default_hf_model()
        
        try:
            logger.info(f"Loading HuggingFace model: {model_name}")
            logger.info(f"Device: {self.device}")
            
            # Initialize tokenizer and model
            self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # Set up generation pipeline for easier use
            self.hf_pipeline = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=self.hf_tokenizer,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            
            self.model_name = model_name
            self.use_ai = True
            logger.info(f"HuggingFace backend initialized successfully with {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace backend: {e}")
            self.hf_pipeline = None
            self.hf_tokenizer = None
    
    def _get_default_hf_model(self) -> str:
        """Get the default HuggingFace model based on available resources."""
        if self.device == "cuda" and torch and torch.cuda.is_available():
            # For GPU, use a more capable instruction-following model
            return "microsoft/DialoGPT-medium"  # Start with smaller model for compatibility
        else:
            # For CPU, use a very lightweight model
            return "microsoft/DialoGPT-small"  # Very lightweight for CPU
    
    def _generate_with_ai(self, prompt: str, max_tokens: int = 1500) -> str:
        """Generate text using the active AI backend."""
        if self.openai_client:
            return self._generate_with_openai(prompt, max_tokens)
        elif self.hf_pipeline:
            return self._generate_with_huggingface(prompt, max_tokens)
        else:
            raise ValueError("No AI backend available")
    
    def _generate_with_openai(self, prompt: str, max_tokens: int = 1500) -> str:
        """Generate text using OpenAI."""
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    
    def _generate_with_huggingface(self, prompt: str, max_tokens: int = 1500) -> str:
        """Generate text using HuggingFace model."""
        # Format prompt for better results with open source models
        formatted_prompt = self._format_prompt_for_hf(prompt)
        
        # Truncate prompt if too long
        max_input_length = 1000  # Leave room for generation
        if len(formatted_prompt.split()) > max_input_length:
            words = formatted_prompt.split()
            formatted_prompt = ' '.join(words[:max_input_length]) + "..."
        
        response = self.hf_pipeline(
            formatted_prompt,
            max_new_tokens=min(max_tokens // 4, 500),  # Limit output tokens for better results
            temperature=0.1,
            do_sample=True,
            pad_token_id=self.hf_tokenizer.eos_token_id,
            truncation=True,
            return_full_text=False  # Only return generated text
        )
        
        # Extract generated text
        generated_text = response[0]['generated_text']
        return generated_text.strip()
    
    def _format_prompt_for_hf(self, prompt: str) -> str:
        """Format prompts to work better with open source models."""
        # Add instruction formatting that works better with open models
        return f"### Instruction:\n{prompt}\n\n### Response:\n"
    
    def build_property_fact_base(self, text: str) -> PropertyFactBase:
        """
        Build a comprehensive fact base for property assets using AI prompts.
        
        Args:
            text (str): Insurance document text
            
        Returns:
            PropertyFactBase: Comprehensive property information
        """
        if not self.use_ai:
            logger.warning("AI not available. Using pattern fallback for limited property extraction.")
            return self._build_property_fact_base_fallback(text)
        
        try:
            prompt = f"""
            You are an expert property inspector and insurance analyst. Analyze this insurance document and extract comprehensive property information to build a detailed fact base.

            Document text:
            {text[:6000]}...

            Extract ALL available information about the property and organize it into the following categories. If information is not explicitly stated, mark as "Not specified" - do not guess.

            Respond with a JSON object with these exact fields:
            {{
                "basic_info": {{
                    "property_type": "Single Family/Condo/Townhouse/etc.",
                    "address": "full address if available",
                    "year_built": "year if mentioned",
                    "square_footage": "size if mentioned"
                }},
                "construction": {{
                    "construction_type": "Frame/Brick/Concrete/etc.",
                    "roof_type": "Asphalt/Tile/Metal/etc.",
                    "roof_age": "age if mentioned",
                    "foundation_type": "Slab/Basement/Crawl Space/etc.",
                    "exterior_material": "Vinyl/Brick/Wood/etc."
                }},
                "interior": {{
                    "flooring_types": ["list of flooring types mentioned"],
                    "heating_system": "Gas/Electric/Oil/etc.",
                    "cooling_system": "Central Air/Window Units/etc.",
                    "electrical_system": "100amp/200amp/etc.",
                    "plumbing_type": "Copper/PVC/etc."
                }},
                "layout": {{
                    "bedrooms": "number if mentioned",
                    "bathrooms": "number if mentioned", 
                    "stories": "number if mentioned",
                    "garage_type": "Attached/Detached/Carport/None",
                    "garage_spaces": "number if mentioned",
                    "basement": "Full/Partial/None",
                    "pool": "In-ground/Above-ground/None"
                }},
                "safety": {{
                    "security_system": "Yes/No/type if mentioned",
                    "fire_detection": "Smoke detectors/Sprinkler system/etc.",
                    "safety_features": ["list any safety features mentioned"]
                }},
                "coverage": {{
                    "dwelling_coverage": "amount if mentioned",
                    "personal_property_coverage": "amount if mentioned",
                    "liability_coverage": "amount if mentioned",
                    "deductible": "amount if mentioned",
                    "additional_coverage": ["list any additional coverages"]
                }},
                "risk_factors": ["list any risk factors mentioned like flood zone, earthquake area, etc."]
            }}
            """
            
            response_text = self._generate_with_ai(prompt, max_tokens=2000)
            
            # Try to extract JSON from the response
            try:
                property_data = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to find JSON in the response text
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    property_data = json.loads(json_match.group())
                else:
                    raise ValueError("No valid JSON found in response")
            
            # Convert to PropertyFactBase
            fact_base = PropertyFactBase(
                # Basic Info
                property_type=property_data.get("basic_info", {}).get("property_type", ""),
                address=property_data.get("basic_info", {}).get("address", ""),
                year_built=property_data.get("basic_info", {}).get("year_built", ""),
                square_footage=property_data.get("basic_info", {}).get("square_footage", ""),
                
                # Construction
                construction_type=property_data.get("construction", {}).get("construction_type", ""),
                roof_type=property_data.get("construction", {}).get("roof_type", ""),
                roof_age=property_data.get("construction", {}).get("roof_age", ""),
                foundation_type=property_data.get("construction", {}).get("foundation_type", ""),
                exterior_material=property_data.get("construction", {}).get("exterior_material", ""),
                
                # Interior
                flooring_types=property_data.get("interior", {}).get("flooring_types", []),
                heating_system=property_data.get("interior", {}).get("heating_system", ""),
                cooling_system=property_data.get("interior", {}).get("cooling_system", ""),
                electrical_system=property_data.get("interior", {}).get("electrical_system", ""),
                plumbing_type=property_data.get("interior", {}).get("plumbing_type", ""),
                
                # Layout
                bedrooms=property_data.get("layout", {}).get("bedrooms", ""),
                bathrooms=property_data.get("layout", {}).get("bathrooms", ""),
                stories=property_data.get("layout", {}).get("stories", ""),
                garage_type=property_data.get("layout", {}).get("garage_type", ""),
                garage_spaces=property_data.get("layout", {}).get("garage_spaces", ""),
                basement=property_data.get("layout", {}).get("basement", ""),
                pool=property_data.get("layout", {}).get("pool", ""),
                
                # Safety
                security_system=property_data.get("safety", {}).get("security_system", ""),
                fire_detection=property_data.get("safety", {}).get("fire_detection", ""),
                safety_features=property_data.get("safety", {}).get("safety_features", []),
                
                # Coverage
                dwelling_coverage=property_data.get("coverage", {}).get("dwelling_coverage", ""),
                personal_property_coverage=property_data.get("coverage", {}).get("personal_property_coverage", ""),
                liability_coverage=property_data.get("coverage", {}).get("liability_coverage", ""),
                deductible=property_data.get("coverage", {}).get("deductible", ""),
                additional_coverage=property_data.get("coverage", {}).get("additional_coverage", []),
                
                # Risk factors
                risk_factors=property_data.get("risk_factors", [])
            )
            
            logger.info("Built comprehensive property fact base using AI analysis")
            return fact_base
            
        except Exception as e:
            logger.error(f"AI property fact base generation failed: {e}. Using fallback.")
            return self._build_property_fact_base_fallback(text)
    
    def build_vehicle_fact_base(self, text: str) -> List[VehicleFactBase]:
        """
        Build comprehensive fact bases for vehicle assets using AI prompts.
        
        Args:
            text (str): Insurance document text
            
        Returns:
            List[VehicleFactBase]: List of comprehensive vehicle information
        """
        if not self.use_ai:
            logger.warning("AI not available. Using pattern fallback for limited vehicle extraction.")
            return self._build_vehicle_fact_base_fallback(text)
        
        try:
            prompt = f"""
            You are an expert automotive specialist and insurance analyst. Analyze this insurance document and extract comprehensive information about all vehicles to build detailed fact bases.

            Document text:
            {text[:6000]}...

            Extract ALL available information about each vehicle and organize it. If information is not explicitly stated, mark as "Not specified" - do not guess.

            Respond with a JSON object:
            {{
                "vehicles": [
                    {{
                        "basic_info": {{
                            "make": "vehicle make",
                            "model": "vehicle model", 
                            "year": "model year",
                            "vin": "VIN if available"
                        }},
                        "specifications": {{
                            "body_type": "Sedan/SUV/Truck/etc.",
                            "fuel_type": "Gasoline/Diesel/Electric/Hybrid",
                            "transmission": "Manual/Automatic/CVT",
                            "engine_size": "engine size if mentioned",
                            "drivetrain": "FWD/RWD/AWD/4WD"
                        }},
                        "details": {{
                            "mileage": "current mileage if mentioned",
                            "color": "color if mentioned",
                            "trim_level": "trim if mentioned",
                            "doors": "number of doors",
                            "seating_capacity": "seating capacity"
                        }},
                        "safety_security": {{
                            "safety_features": ["list safety features mentioned"],
                            "security_features": ["list security features mentioned"]
                        }},
                        "usage": {{
                            "primary_use": "Personal/Business/Farm/etc.",
                            "annual_mileage": "annual mileage if mentioned",
                            "garage_kept": "Yes/No/Sometimes",
                            "condition": "condition if mentioned"
                        }},
                        "modifications": {{
                            "modifications": ["list any modifications"],
                            "aftermarket_parts": ["list aftermarket parts"]
                        }},
                        "coverage": {{
                            "policy_number": "policy number if available",
                            "coverage_type": "coverage type",
                            "liability_limit": "liability limits",
                            "comprehensive_deductible": "comprehensive deductible",
                            "collision_deductible": "collision deductible",
                            "additional_coverage": ["additional coverages"]
                        }},
                        "risk_factors": ["any risk factors mentioned"]
                    }}
                ]
            }}
            """
            
            response_text = self._generate_with_ai(prompt, max_tokens=2500)
            
            # Try to extract JSON from the response
            try:
                vehicles_data = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to find JSON in the response text
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    vehicles_data = json.loads(json_match.group())
                else:
                    raise ValueError("No valid JSON found in vehicle response")
            
            # Convert to VehicleFactBase objects
            fact_bases = []
            for vehicle_data in vehicles_data.get("vehicles", []):
                fact_base = VehicleFactBase(
                    # Basic Info
                    make=vehicle_data.get("basic_info", {}).get("make", ""),
                    model=vehicle_data.get("basic_info", {}).get("model", ""),
                    year=vehicle_data.get("basic_info", {}).get("year", ""),
                    vin=vehicle_data.get("basic_info", {}).get("vin", ""),
                    
                    # Specifications
                    body_type=vehicle_data.get("specifications", {}).get("body_type", ""),
                    fuel_type=vehicle_data.get("specifications", {}).get("fuel_type", ""),
                    transmission=vehicle_data.get("specifications", {}).get("transmission", ""),
                    engine_size=vehicle_data.get("specifications", {}).get("engine_size", ""),
                    drivetrain=vehicle_data.get("specifications", {}).get("drivetrain", ""),
                    
                    # Details
                    mileage=vehicle_data.get("details", {}).get("mileage", ""),
                    color=vehicle_data.get("details", {}).get("color", ""),
                    trim_level=vehicle_data.get("details", {}).get("trim_level", ""),
                    doors=vehicle_data.get("details", {}).get("doors", ""),
                    seating_capacity=vehicle_data.get("details", {}).get("seating_capacity", ""),
                    
                    # Safety & Security
                    safety_features=vehicle_data.get("safety_security", {}).get("safety_features", []),
                    security_features=vehicle_data.get("safety_security", {}).get("security_features", []),
                    
                    # Usage
                    primary_use=vehicle_data.get("usage", {}).get("primary_use", ""),
                    annual_mileage=vehicle_data.get("usage", {}).get("annual_mileage", ""),
                    garage_kept=vehicle_data.get("usage", {}).get("garage_kept", ""),
                    condition=vehicle_data.get("usage", {}).get("condition", ""),
                    
                    # Modifications
                    modifications=vehicle_data.get("modifications", {}).get("modifications", []),
                    aftermarket_parts=vehicle_data.get("modifications", {}).get("aftermarket_parts", []),
                    
                    # Coverage
                    policy_number=vehicle_data.get("coverage", {}).get("policy_number", ""),
                    coverage_type=vehicle_data.get("coverage", {}).get("coverage_type", ""),
                    liability_limit=vehicle_data.get("coverage", {}).get("liability_limit", ""),
                    comprehensive_deductible=vehicle_data.get("coverage", {}).get("comprehensive_deductible", ""),
                    collision_deductible=vehicle_data.get("coverage", {}).get("collision_deductible", ""),
                    additional_coverage=vehicle_data.get("coverage", {}).get("additional_coverage", []),
                    
                    # Risk factors
                    risk_factors=vehicle_data.get("risk_factors", [])
                )
                fact_bases.append(fact_base)
            
            logger.info(f"Built {len(fact_bases)} comprehensive vehicle fact bases using AI analysis")
            return fact_bases
            
        except Exception as e:
            logger.error(f"AI vehicle fact base generation failed: {e}. Using fallback.")
            return self._build_vehicle_fact_base_fallback(text)
    
    def analyze_document_with_fact_bases(self, text: str) -> Dict[str, Any]:
        """
        Analyze insurance document and build comprehensive fact bases for all assets.
        
        Args:
            text (str): Insurance document text
            
        Returns:
            Dict[str, Any]: Comprehensive analysis with detailed fact bases
        """
        logger.info("Starting comprehensive fact-based document analysis")
        start_time = datetime.now()
        
        # Build fact bases using AI prompts
        property_fact_base = self.build_property_fact_base(text)
        vehicle_fact_bases = self.build_vehicle_fact_base(text)
        
        # Extract policy information (can reuse existing method)
        if hasattr(self, 'extract_policy_info'):
            policy_info = self.extract_policy_info(text)
        else:
            policy_info = PolicySummary()
        
        # Create comprehensive results
        processing_time = (datetime.now() - start_time).total_seconds()
        metadata = AnalysisMetadata(
            analysis_date=datetime.now().isoformat(),
            total_vehicles=len(vehicle_fact_bases),
            total_properties=1 if property_fact_base.address or property_fact_base.property_type else 0,
            document_length=len(text),
            processing_time_seconds=processing_time,
            confidence_score=0.95 if self.use_ai else 0.7  # Higher confidence for AI analysis
        )
        
        results = {
            'analysis_type': 'comprehensive_fact_based',
            'policy_summary': asdict(policy_info),
            'property_fact_base': asdict(property_fact_base),
            'vehicle_fact_bases': [asdict(vehicle) for vehicle in vehicle_fact_bases],
            'analysis_metadata': asdict(metadata),
            'ai_powered': self.use_ai
        }
        
        logger.info(f"Fact-based analysis complete - {len(vehicle_fact_bases)} vehicles, {metadata.total_properties} properties")
        return results
    
    def _build_property_fact_base_fallback(self, text: str) -> PropertyFactBase:
        """Fallback method using pattern matching for basic property info."""
        # Basic fallback implementation
        fact_base = PropertyFactBase()
        if hasattr(self, 'extract_properties') and self.fallback_patterns:
            properties = self.extract_properties(text)
            if properties:
                prop = properties[0]  # Take first property
                fact_base.property_type = getattr(prop, 'property_type', '')
                fact_base.address = getattr(prop, 'address', '')
                fact_base.construction_type = getattr(prop, 'construction_type', '')
                fact_base.year_built = getattr(prop, 'year_built', '')
                fact_base.square_footage = getattr(prop, 'square_footage', '')
                fact_base.dwelling_coverage = getattr(prop, 'dwelling_coverage', '')
                fact_base.personal_property_coverage = getattr(prop, 'personal_property_coverage', '')
                fact_base.liability_coverage = getattr(prop, 'liability_coverage', '')
                fact_base.deductible = getattr(prop, 'deductible', '')
        return fact_base
    
    def _build_vehicle_fact_base_fallback(self, text: str) -> List[VehicleFactBase]:
        """Fallback method using pattern matching for basic vehicle info."""
        fact_bases = []
        if hasattr(self, 'extract_vehicles') and self.fallback_patterns:
            vehicles = self.extract_vehicles(text)
            for vehicle in vehicles:
                fact_base = VehicleFactBase()
                fact_base.make = getattr(vehicle, 'make', '')
                fact_base.model = getattr(vehicle, 'model', '')
                fact_base.year = getattr(vehicle, 'year', '')
                fact_base.vin = getattr(vehicle, 'vin', '')
                fact_base.coverage_type = getattr(vehicle, 'coverage_type', '')
                fact_base.liability_limit = getattr(vehicle, 'liability_limit', '')
                fact_base.comprehensive_deductible = getattr(vehicle, 'comprehensive_deductible', '')
                fact_base.collision_deductible = getattr(vehicle, 'collision_deductible', '')
                fact_bases.append(fact_base)
        return fact_bases
    
    def _init_vehicle_patterns(self) -> Dict[str, str]:
        """Initialize regex patterns for vehicle information extraction."""
        return {
            # Updated patterns for better accuracy
            'vehicle_line': r'(\d{4})\s+([A-Z]+)\s+([A-Z0-9\s]+)\s+([A-HJ-NPR-Z0-9]{17})',  # Year Make Model VIN
            'vehicle_detail': r'(\d{2})\s+([A-Z]+)\s+([A-Z0-9\s]+)',  # Year Make Model (shortened format)
            'vin': r'([A-HJ-NPR-Z0-9]{17})',  # Just VIN pattern
            'year': r'(\d{4})',  # 4-digit year
            'bodily_injury': r'(\$[\d,]+)\s+each\s+person.*?(\$[\d,]+)\s+each\s+accident.*?(\$[\d,]+).*?(\$[\d,]+)',
            'property_damage': r'Property Damage.*?(\$[\d,]+)\s+each\s+accident.*?(\$[\d,]+).*?(\$[\d,]+)',
            'collision_deductible': r'Collision.*?(\$[\d,]+)\s+deductible.*?(\$[\d,]+).*?(\$[\d,]+)',
            'comprehensive_deductible': r'Comprehensive.*?(\$[\d,]+)\s+deductible.*?(\$[\d,]+).*?(\$[\d,]+)',
            'vehicle_location': r'([A-Z\s]+),\s+([A-Z]{2})',  # City, State
        }
    
    def _init_property_patterns(self) -> Dict[str, str]:
        """Initialize regex patterns for property information extraction."""
        return {
            'property_type': r'(?:Property Type[:\s]+|Dwelling Type[:\s]+)(Single Family|Condo|Condominium|Townhouse|Apartment|Mobile Home|Manufactured Home)',
            'address': r'(?:Property Address[:\s]+|Address[:\s]+)([A-Za-z0-9\s,.-]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Boulevard|Blvd|Lane|Ln|Circle|Cir|Court|Ct)[A-Za-z0-9\s,.-]*)',
            'construction': r'(?:Construction[:\s]+|Construction Type[:\s]+)(Frame|Brick|Concrete|Steel|Masonry|Wood Frame)',
            'year_built': r'(?:Year Built[:\s]+|Built[:\s]+)(\d{4})',
            'square_feet': r'(?:Square Feet[:\s]+|Sq\.?\s*Ft\.?[:\s]+|Square Footage[:\s]+)([\d,]+)',
            'dwelling_coverage': r'(?:Dwelling Coverage[:\s]+|Coverage A[:\s]+)(\$[\d,]+)',
            'personal_property': r'(?:Personal Property[:\s]+|Coverage C[:\s]+)(\$[\d,]+)',
            'liability': r'(?:Personal Liability[:\s]+|Coverage E[:\s]+)(\$[\d,]+)',
            'deductible': r'(?:Deductible[:\s]+|All Other Perils[:\s]+)(\$[\d,]+)',
        }
    
    def _init_policy_patterns(self) -> Dict[str, str]:
        """Initialize regex patterns for policy information extraction."""
        return {
            'policy_number': r'(?:Policy Number|Auto Policy Number)[:\s]+([A-Z0-9\s]+)',
            'policy_holder': r'(?:Named Insured|1\.\s*Named Insured)\s*(?:Your Agency\'s Name and Address\s*)?\n?\s*([A-Z\s]+AND\s+[A-Z\s]+)',
            'effective_date': r'(\d{2}\/\d{2}\/\d{4})\s+to\s+(\d{2}\/\d{2}\/\d{4})',
            'total_premium': r'Total Premium for this Policy:\s*(\$[\d,]+)',
            'savings': r'Your Total Savings.*?(\$[\d,]+)',
            'carrier': r'(THE STANDARD FIRE INSURANCE COMPANY|[A-Z\s&]+INSURANCE[A-Z\s]*)',
            'agent_company': r'([A-Z\s]+(?:INS|INSURANCE)\s+SERVICES)',
            'address': r'(\d+[A-Z\s]+(?:PL|ST|AVE|RD|DR|BLVD|LN|CT)\s*\n?[A-Z\s,]+\d{5}(?:-\d{4})?)',
        }
    
    def _init_coverage_patterns(self) -> Dict[str, str]:
        """Initialize regex patterns for coverage information."""
        return {
            'flood_coverage': r'(?:Flood[:\s]+|Flood Insurance[:\s]+)(\$[\d,]+|Yes|No)',
            'earthquake_coverage': r'(?:Earthquake[:\s]+|Earthquake Coverage[:\s]+)(\$[\d,]+|Yes|No)',
            'umbrella_coverage': r'(?:Umbrella[:\s]+|Umbrella Policy[:\s]+)(\$[\d,]+)',
            'jewelry_coverage': r'(?:Jewelry[:\s]+|Scheduled Jewelry[:\s]+)(\$[\d,]+)',
        }
    
    def extract_vehicles(self, text: str) -> List[Vehicle]:
        """
        Extract vehicle information from insurance document text.
        
        Args:
            text (str): Raw text from insurance document
            
        Returns:
            List[Vehicle]: List of identified vehicles
        """
        vehicles = []
        
        # Look for vehicle identification lines like "2017 TOYOT HIGHLANDER 5TDZZRFH3HS198659"
        vehicle_line_matches = re.findall(self.vehicle_patterns['vehicle_line'], text)
        
        for match in vehicle_line_matches:
            vehicle = Vehicle()
            vehicle.year = match[0]
            # Expand truncated make names using AI or fallback to pattern matching
            make = match[1]
            vehicle.make = self._correct_vehicle_make(make)
            vehicle.model = match[2].strip()
            vehicle.vin = match[3]
            vehicles.append(vehicle)
        
        # If we didn't find the full pattern, try to find VINs and match them with other info
        if not vehicles:
            vin_matches = re.findall(self.vehicle_patterns['vin'], text)
            for vin in vin_matches:
                vehicle = Vehicle()
                vehicle.vin = vin
                
                # Try to find associated vehicle info around this VIN
                vin_position = text.find(vin)
                if vin_position != -1:
                    # Look in surrounding text for year/make/model
                    surrounding_text = text[max(0, vin_position-200):vin_position+200]
                    
                    # Look for shortened format like "17 TOYOT" or "18 PORSE"
                    detail_match = re.search(self.vehicle_patterns['vehicle_detail'], surrounding_text)
                    if detail_match:
                        vehicle.year = "20" + detail_match.group(1)  # Convert 17 to 2017
                        vehicle.make = self._correct_vehicle_make(detail_match.group(2))
                        vehicle.model = detail_match.group(3).strip()
                
                vehicles.append(vehicle)
        
        return vehicles
    
    def extract_properties(self, text: str) -> List[Property]:
        """
        Extract property/home information from insurance document text.
        
        Args:
            text (str): Raw text from insurance document
            
        Returns:
            List[Property]: List of identified properties
        """
        properties = []
        
        # Check if this is actually a property/home insurance document
        # Auto policies don't typically have property coverage
        if "Automobile Policy" in text or "Auto Policy" in text:
            # This is an auto policy, likely no property coverage
            return properties
        
        # Find property sections in the text
        property_sections = self._find_property_sections(text)
        
        for section in property_sections:
            prop = Property()
            
            # Extract property type
            prop_type_match = re.search(self.property_patterns['property_type'], section, re.IGNORECASE)
            if prop_type_match:
                prop.property_type = prop_type_match.group(1)
            
            # Extract address
            address_match = re.search(self.property_patterns['address'], section, re.IGNORECASE)
            if address_match:
                prop.address = address_match.group(1).strip()
            
            # Extract construction type
            construction_match = re.search(self.property_patterns['construction'], section, re.IGNORECASE)
            if construction_match:
                prop.construction_type = construction_match.group(1)
            
            # Extract year built
            year_match = re.search(self.property_patterns['year_built'], section, re.IGNORECASE)
            if year_match:
                prop.year_built = year_match.group(1)
            
            # Extract square footage
            sqft_match = re.search(self.property_patterns['square_feet'], section, re.IGNORECASE)
            if sqft_match:
                prop.square_footage = sqft_match.group(1)
            
            # Extract coverage amounts
            dwelling_match = re.search(self.property_patterns['dwelling_coverage'], section, re.IGNORECASE)
            if dwelling_match:
                prop.dwelling_coverage = dwelling_match.group(1)
            
            personal_prop_match = re.search(self.property_patterns['personal_property'], section, re.IGNORECASE)
            if personal_prop_match:
                prop.personal_property_coverage = personal_prop_match.group(1)
            
            liability_match = re.search(self.property_patterns['liability'], section, re.IGNORECASE)
            if liability_match:
                prop.liability_coverage = liability_match.group(1)
            
            deductible_match = re.search(self.property_patterns['deductible'], section, re.IGNORECASE)
            if deductible_match:
                prop.deductible = deductible_match.group(1)
            
            # Check for additional coverages
            for coverage_name, pattern in self.coverage_patterns.items():
                match = re.search(pattern, section, re.IGNORECASE)
                if match:
                    prop.additional_coverage.append(f"{coverage_name}: {match.group(1)}")
            
            # Only add property if we found meaningful property information
            if prop.property_type or prop.dwelling_coverage:
                properties.append(prop)
        
        return properties
    
    def extract_policy_info(self, text: str) -> PolicySummary:
        """
        Extract general policy information from insurance document text.
        
        Args:
            text (str): Raw text from insurance document
            
        Returns:
            PolicySummary: Policy information
        """
        policy = PolicySummary()
        
        # Extract policy number
        policy_num_match = re.search(self.policy_patterns['policy_number'], text)
        if policy_num_match:
            policy.policy_number = policy_num_match.group(1).strip()
        
        # Extract policy holder
        holder_match = re.search(self.policy_patterns['policy_holder'], text)
        if holder_match:
            policy.policy_holder = holder_match.group(1).strip()
        
        # Extract policy period (effective and expiration dates)
        date_match = re.search(self.policy_patterns['effective_date'], text)
        if date_match:
            policy.effective_date = date_match.group(1)
            policy.expiration_date = date_match.group(2)
        
        # Extract total premium
        premium_match = re.search(self.policy_patterns['total_premium'], text)
        if premium_match:
            policy.premium = premium_match.group(1)
        
        # Extract savings if available
        savings_match = re.search(self.policy_patterns['savings'], text)
        if savings_match:
            policy.premium += f" (Savings: {savings_match.group(1)})"
        
        # Extract carrier
        carrier_match = re.search(self.policy_patterns['carrier'], text)
        if carrier_match:
            policy.carrier = carrier_match.group(1).strip()
        
        # Extract agent/agency
        agent_match = re.search(self.policy_patterns['agent_company'], text)
        if agent_match:
            policy.agent = agent_match.group(1).strip()
        
        return policy
    
    def _find_property_sections(self, text: str) -> List[str]:
        """Find sections of text that likely contain property information."""
        sections = []
        
        # Look for home/property sections
        property_section_pattern = r'(?:HOME|PROPERTY|DWELLING|RESIDENCE).*?(?=(?:AUTO|VEHICLE|SCHEDULE|PREMIUM|TOTAL|\Z))'
        matches = re.findall(property_section_pattern, text, re.IGNORECASE | re.DOTALL)
        sections.extend(matches)
        
        # Also look for lines that contain property keywords
        lines = text.split('\n')
        property_keywords = ['dwelling', 'property', 'home', 'residence', 'address', 'coverage a', 'coverage c']
        
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in property_keywords):
                # Take this line and a few around it for context
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                section = '\n'.join(lines[start:end])
                if section not in sections:
                    sections.append(section)
        
        return sections
    
    def _correct_vehicle_make(self, make: str) -> str:
        """
        Correct vehicle make names using AI or fallback to pattern matching.
        
        Args:
            make (str): Raw vehicle make name (potentially truncated)
            
        Returns:
            str: Corrected vehicle make name
        """
        if not make:
            return make
        
        # If AI is available, use it for intelligent correction
        if self.use_ai and self.openai_client:
            try:
                prompt = f"""
                You are an automotive expert. I have a vehicle make name that appears to be truncated or abbreviated from an insurance document: "{make}"
                
                Please provide the correct full vehicle manufacturer name. Common truncations include:
                - TOYOT → TOYOTA
                - PORSE → PORSCHE  
                - MERCD → MERCEDES
                - VOLKS → VOLKSWAGEN
                - CHEV → CHEVROLET
                - etc.
                
                Respond with just the corrected make name, nothing else. If the name is already correct or you're not sure, return it unchanged.
                """
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=50
                )
                
                corrected_make = response.choices[0].message.content.strip().upper()
                logger.info(f"AI corrected vehicle make: '{make}' → '{corrected_make}'")
                return corrected_make
                
            except Exception as e:
                logger.warning(f"AI make correction failed: {e}. Using fallback.")
        
        # Fallback to common known corrections if AI is not available
        make_corrections = {
            "TOYOT": "TOYOTA",
            "PORSE": "PORSCHE", 
            "MERCD": "MERCEDES",
            "MERCS": "MERCEDES",
            "VOLKS": "VOLKSWAGEN",
            "CHEV": "CHEVROLET",
            "CHRYS": "CHRYSLER",
            "INFINIT": "INFINITI",
            "MITSU": "MITSUBISHI",
            "SUBAR": "SUBARU",
            "ACUR": "ACURA",
            "LEXU": "LEXUS",
            "CADIL": "CADILLAC",
            "LINCO": "LINCOLN",
            "JAGUA": "JAGUAR",
            "LAMBO": "LAMBORGHINI",
            "MASR": "MASERATI",
            "FERR": "FERRARI",
            "BENT": "BENTLEY",
            "ROLL": "ROLLS-ROYCE"
        }
        
        corrected = make_corrections.get(make.upper(), make)
        if corrected != make:
            logger.info(f"Fallback corrected vehicle make: '{make}' → '{corrected}'")
        
        return corrected
    
    def analyze_pdf_file(self, filename: str, use_advanced: bool = True) -> Dict[str, Any]:
        """
        Directly analyze a PDF file by first extracting text, then performing analysis.
        
        Args:
            filename (str): Name of the PDF file to analyze
            use_advanced (bool): Whether to use advanced PDF extraction
            
        Returns:
            Dict[str, Any]: Complete analysis results including PDF metadata
        """
        logger.info(f"Starting direct PDF analysis: {filename}")
        
        # Step 1: Extract text and metadata from PDF
        pdf_data = self.pdf_reader.process_pdf(filename, use_advanced)
        
        if not pdf_data:
            logger.error(f"Failed to extract data from PDF: {filename}")
            return {}
        
        # Step 2: Perform document analysis on extracted text
        text_content = pdf_data.get('text', '')
        analysis_results = self.analyze_document_with_fact_bases(text_content)
        
        # Step 3: Merge PDF metadata with analysis results
        analysis_results['pdf_metadata'] = pdf_data.get('metadata', {})
        analysis_results['pdf_tables'] = pdf_data.get('tables', [])
        analysis_results['source_file'] = filename
        
        logger.info(f"Direct PDF analysis complete for: {filename}")
        return analysis_results
    
    def save_analysis(self, analysis_results: Dict[str, Any], output_path: str) -> str:
        """
        Save analysis results to a JSON file.
        
        Args:
            analysis_results (Dict): Results from analyze_document()
            output_path (str): Path to save the results
            
        Returns:
            str: Path to saved file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis results saved to: {output_file}")
        return str(output_file)


def main():
    """
    Demonstration of the Prompt-Based Asset Researcher with comprehensive fact base generation.
    """
    print("🏠🚗 PROMPT-BASED ASSET RESEARCHER - COMPREHENSIVE FACT BASE ANALYSIS")
    print("=" * 80)
    
    # Initialize the new prompt-based researcher with multi-backend support
    try:
        print("🤖 Initializing AI backend (trying auto-selection)...")
        researcher = PromptBasedAssetResearcher(model_type="auto", fallback_patterns=True)
    except ValueError as e:
        print(f"❌ Initialization failed: {e}")
        print("Please set OPENAI_API_KEY environment variable or install transformers for full functionality.")
        return
    
    # Check for PDF files to process directly
    pdf_files = list(researcher.pdf_reader.input_dir.glob("*.pdf"))
    text_files = list(researcher.pdf_reader.output_dir.glob("*_extracted.txt"))
    
    if not pdf_files and not text_files:
        print(f"\n❌ No PDF or text files found.")
        print(f"Place PDF files in: {researcher.pdf_reader.input_dir}")
        print("Or ensure extracted text files exist in: output/")
        return
    
    # Determine what to analyze
    if pdf_files:
        print(f"\n✅ Found {len(pdf_files)} PDF file(s) for direct processing:")
        for pdf_file in pdf_files:
            print(f"  📄 {pdf_file.name}")
        analysis_file = pdf_files[0]
        
        # Extract text from PDF first
        print(f"\n📖 Extracting text from: {analysis_file.name}")
        pdf_data = researcher.pdf_reader.process_pdf(analysis_file.name)
        document_text = pdf_data.get('text', '') if pdf_data else ''
        
    else:
        print(f"\n✅ Found {len(text_files)} extracted text file(s):")
        for text_file in text_files:
            print(f"  📄 {text_file.name}")
        
        # Use existing extracted text
        with open(text_files[0], 'r', encoding='utf-8') as f:
            document_text = f.read()
    
    if not document_text:
        print("❌ No text content available for analysis")
        return
    
    # Demonstrate new fact-based analysis
    print("\n" + "=" * 80)
    print("🔬 COMPREHENSIVE FACT-BASED ANALYSIS")
    print("=" * 80)
    
    print(f"\n🤖 AI-powered analysis: {'✅ ENABLED' if researcher.use_ai else '❌ DISABLED (using patterns)'}")
    print(f"📄 Document length: {len(document_text):,} characters")
    
    # Perform comprehensive fact-based analysis
    print(f"\n🏗️ Building comprehensive asset fact bases...")
    fact_results = researcher.analyze_document_with_fact_bases(document_text)
    
    # Display property fact base
    print(f"\n🏠 PROPERTY FACT BASE:")
    print("-" * 40)
    prop_facts = fact_results.get('property_fact_base', {})
    
    if prop_facts.get('address') or prop_facts.get('property_type'):
        print(f"📍 Address: {prop_facts.get('address', 'Not specified')}")
        print(f"🏘️  Type: {prop_facts.get('property_type', 'Not specified')}")
        print(f"📅 Year Built: {prop_facts.get('year_built', 'Not specified')}")
        print(f"📐 Square Footage: {prop_facts.get('square_footage', 'Not specified')}")
        
        print(f"\n🔨 Construction Details:")
        print(f"  Construction: {prop_facts.get('construction_type', 'Not specified')}")
        print(f"  Roof Type: {prop_facts.get('roof_type', 'Not specified')}")
        print(f"  Foundation: {prop_facts.get('foundation_type', 'Not specified')}")
        print(f"  Exterior: {prop_facts.get('exterior_material', 'Not specified')}")
        
        print(f"\n🏠 Layout & Features:")
        print(f"  Bedrooms: {prop_facts.get('bedrooms', 'Not specified')}")
        print(f"  Bathrooms: {prop_facts.get('bathrooms', 'Not specified')}")
        print(f"  Garage: {prop_facts.get('garage_type', 'Not specified')}")
        print(f"  Pool: {prop_facts.get('pool', 'Not specified')}")
        
        flooring = prop_facts.get('flooring_types', [])
        if flooring:
            print(f"  Flooring: {', '.join(flooring)}")
    else:
        print("❌ No property information found in document")
    
    # Display vehicle fact bases
    print(f"\n🚗 VEHICLE FACT BASES:")
    print("-" * 40)
    vehicle_facts = fact_results.get('vehicle_fact_bases', [])
    
    if vehicle_facts:
        for i, vehicle in enumerate(vehicle_facts, 1):
            print(f"\n🚗 Vehicle {i}:")
            print(f"  Make/Model: {vehicle.get('make', '')} {vehicle.get('model', '')}")
            print(f"  Year: {vehicle.get('year', 'Not specified')}")
            print(f"  VIN: {vehicle.get('vin', 'Not specified')}")
            
            # Specifications
            body_type = vehicle.get('body_type', 'Not specified')
            fuel_type = vehicle.get('fuel_type', 'Not specified')
            if body_type != 'Not specified' or fuel_type != 'Not specified':
                print(f"  Specifications:")
                print(f"    Body Type: {body_type}")
                print(f"    Fuel Type: {fuel_type}")
                print(f"    Transmission: {vehicle.get('transmission', 'Not specified')}")
            
            # Safety features
            safety_features = vehicle.get('safety_features', [])
            if safety_features:
                print(f"    Safety: {', '.join(safety_features)}")
            
            # Coverage
            liability = vehicle.get('liability_limit', 'Not specified')
            if liability != 'Not specified':
                print(f"  Coverage: {liability}")
    else:
        print("❌ No vehicle information found in document")
    
    # Save comprehensive results
    output_file = researcher.pdf_reader.output_dir / "comprehensive_fact_analysis.json"
    researcher.save_analysis(fact_results, output_file)
    
    # Analysis summary
    metadata = fact_results.get('analysis_metadata', {})
    print(f"\n📊 ANALYSIS SUMMARY:")
    print("-" * 30)
    print(f"🕒 Processing Time: {metadata.get('processing_time_seconds', 0):.2f} seconds")
    print(f"🎯 Confidence Score: {metadata.get('confidence_score', 0):.0%}")
    print(f"🚗 Vehicles Found: {metadata.get('total_vehicles', 0)}")
    print(f"🏠 Properties Found: {metadata.get('total_properties', 0)}")
    print(f"📄 Analysis Type: {fact_results.get('analysis_type', 'Unknown')}")
    
    print(f"\n💾 Comprehensive results saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("🎉 FACT-BASED ANALYSIS COMPLETE!")
    print("=" * 80)
    print("✨ New capabilities demonstrated:")
    print("  🏠 Comprehensive property fact bases (roof, layout, construction)")
    print("  🚗 Detailed vehicle specifications (body type, fuel, safety)")
    print("  🧠 AI-powered intelligent extraction from any insurance document")
    print("  📊 Structured fact bases for automated processing")
    print("  🎯 High-confidence asset intelligence")
    print("  ⚡ Works with any insurance document type")


if __name__ == "__main__":
    main()
