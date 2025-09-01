#!/usr/bin/env python3
"""
Insurance Data Models Package

Consolidated data models for the insurance system, including:
- Vehicle, Property, and Personal Item models with comprehensive information
- Policy and analysis metadata models
- Insurance card processing models
- Web research result models
- Orchestrator workflow models (leads, bids, world state)
- Factory functions for creating model instances
- Collection management utilities

This package serves as the single source of truth for all data structures
used throughout the insurance system.
"""

# Import all model classes
from .vehicle_models import Vehicle
from .property_models import Property
from .personal_models import PersonalItem
from .policy_models import PolicySummary, AnalysisMetadata
from .card_models import InsuranceCardInfo
from .web_models import SearchResult, WebContent
from .collection_models import AssetCollection
from .orchestrator_models import Lead, BidIntent, BidResult, WorldState
from .factory_functions import (
    create_vehicle_from_dict,
    create_property_from_dict,
    create_personal_item_from_dict,
    create_policy_summary_from_dict,
    validate_vehicle_data,
    validate_property_data,
    sanitize_data
)

# Compatibility aliases for schema.py migration
# These maintain backward compatibility with existing orchestrator code
PolicyInfo = PolicySummary  # PolicyInfo -> PolicySummary
AssetVehicle = Vehicle      # AssetVehicle -> Vehicle  
AssetProperty = Property    # AssetProperty -> Property
AssetPersonal = PersonalItem # AssetPersonal -> PersonalItem

# Export all public classes and functions
__all__ = [
    # Core asset models
    'Vehicle', 'Property', 'PersonalItem',
    
    # Policy and analysis models
    'PolicySummary', 'AnalysisMetadata',
    
    # Insurance card models
    'InsuranceCardInfo',
    
    # Web research models
    'SearchResult', 'WebContent',
    
    # Collection management
    'AssetCollection',
    
    # Orchestrator workflow models
    'Lead', 'BidIntent', 'BidResult', 'WorldState',
    
    # Factory functions
    'create_vehicle_from_dict', 'create_property_from_dict',
    'create_personal_item_from_dict', 'create_policy_summary_from_dict',
    'validate_vehicle_data', 'validate_property_data', 'sanitize_data',
    
    # Compatibility aliases (deprecated, use new names)
    'PolicyInfo', 'AssetVehicle', 'AssetProperty', 'AssetPersonal'
]
