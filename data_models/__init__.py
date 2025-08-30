#!/usr/bin/env python3
"""
Insurance Data Models

This package contains all data models and dataclasses for the insurance analysis system.
Organized by domain for better maintainability and separation of concerns.
"""

# Vehicle models
from .vehicle_models import Vehicle

# Property models
from .property_models import Property

# Policy and analysis models
from .policy_models import PolicySummary, AnalysisMetadata

# Personal items models
from .personal_models import PersonalItem

# Insurance card models
from .card_models import InsuranceCardInfo

# Collection and utility models
from .collection_models import AssetCollection

# Web research models
from .web_models import SearchResult, WebContent

# Utility functions
from .factory_functions import (
    create_vehicle_from_dict,
    create_property_from_dict,
    create_personal_item_from_dict,
    create_policy_summary_from_dict
)

__all__ = [
    # Vehicle models
    'Vehicle',
    
    # Property models
    'Property',
    
    # Policy models
    'PolicySummary',
    'AnalysisMetadata',
    
    # Personal models
    'PersonalItem',
    
    # Card models
    'InsuranceCardInfo',
    
    # Collection models
    'AssetCollection',
    
    # Web models
    'SearchResult',
    'WebContent',
    
    # Factory functions
    'create_vehicle_from_dict',
    'create_property_from_dict',
    'create_personal_item_from_dict',
    'create_policy_summary_from_dict'
]
