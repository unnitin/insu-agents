#!/usr/bin/env python3
"""
Insurance Base Agent

Base class for AI-powered insurance agents providing common backend initialization
and text generation capabilities. Supports OpenAI and HuggingFace backends with
automatic selection and graceful fallback.
"""

import os
import logging
from typing import Optional
from pathlib import Path

# Setup logging
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


class InsuranceBaseAgent:
    """
    Base class for AI-powered insurance agents.
    
    Provides common functionality for backend initialization (OpenAI, HuggingFace),
    text generation, and AI model management. Other insurance agents can inherit
    from this class to get AI capabilities without reimplementing backend logic.
    """
    
    def __init__(self, 
                 model_type: str = "auto",
                 model_name: Optional[str] = None,
                 api_key: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize the base agent with AI backend configuration.
        
        Args:
            model_type (str): Model backend to use ("openai", "huggingface", or "auto")
            model_name (str, optional): Specific model name to use
            api_key (str, optional): OpenAI API key. If not provided, will check environment
            device (str, optional): Device for HuggingFace models ("cpu", "cuda", "auto")
        """
        self.model_type = model_type
        self.model_name = model_name
        self.device = device or ("cuda" if torch and torch.cuda.is_available() else "cpu")
        
        # Initialize model backends
        self.openai_client = None
        self.hf_model = None
        self.hf_tokenizer = None
        self.hf_pipeline = None
        self.use_ai = False
        
        # Initialize AI backend based on model_type
        if model_type == "auto":
            self._initialize_auto_backend(api_key)
        elif model_type == "openai":
            self._initialize_openai_backend(api_key)
        elif model_type == "huggingface":
            self._initialize_huggingface_backend(model_name)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Use 'openai', 'huggingface', or 'auto'")
        
        active_backend = "OpenAI" if self.openai_client else "HuggingFace" if self.hf_pipeline else "None"
        logger.info(f"InsuranceBaseAgent initialized (Backend: {active_backend}, Device: {self.device})")
    
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
        
        logger.warning("No AI backend available. AI features will be disabled.")
    
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
    
    def is_ai_available(self) -> bool:
        """Check if AI backend is available and working."""
        return self.use_ai
    
    def get_backend_info(self) -> dict:
        """Get information about the active backend."""
        return {
            'use_ai': self.use_ai,
            'model_type': self.model_type,
            'model_name': self.model_name,
            'device': self.device,
            'has_openai': bool(self.openai_client),
            'has_huggingface': bool(self.hf_pipeline),
            'backend': "OpenAI" if self.openai_client else "HuggingFace" if self.hf_pipeline else "None"
        }
