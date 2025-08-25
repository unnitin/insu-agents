#!/usr/bin/env python3
"""
Factory Functions for Data Models

Contains utility functions for creating data model instances from dictionary data.
"""

from typing import Dict, Any
from .vehicle_models import Vehicle
from .property_models import Property
from .personal_models import PersonalItem
from .policy_models import PolicySummary


def create_vehicle_from_dict(data: Dict[str, Any]) -> Vehicle:
    """
    Create Vehicle instance from dictionary data.
    
    Args:
        data: Dictionary containing vehicle data
        
    Returns:
        Vehicle instance with populated fields
    """
    return Vehicle(**{k: v for k, v in data.items() if k in Vehicle.__annotations__})


def create_property_from_dict(data: Dict[str, Any]) -> Property:
    """
    Create Property instance from dictionary data.
    
    Args:
        data: Dictionary containing property data
        
    Returns:
        Property instance with populated fields
    """
    return Property(**{k: v for k, v in data.items() if k in Property.__annotations__})


def create_personal_item_from_dict(data: Dict[str, Any]) -> PersonalItem:
    """
    Create PersonalItem instance from dictionary data.
    
    Args:
        data: Dictionary containing personal item data
        
    Returns:
        PersonalItem instance with populated fields
    """
    return PersonalItem(**{k: v for k, v in data.items() if k in PersonalItem.__annotations__})


def create_policy_summary_from_dict(data: Dict[str, Any]) -> PolicySummary:
    """
    Create PolicySummary instance from dictionary data.
    
    Args:
        data: Dictionary containing policy summary data
        
    Returns:
        PolicySummary instance with populated fields
    """
    return PolicySummary(**{k: v for k, v in data.items() if k in PolicySummary.__annotations__})


def validate_vehicle_data(data: Dict[str, Any]) -> bool:
    """
    Validate that dictionary contains minimum required vehicle data.
    
    Args:
        data: Dictionary to validate
        
    Returns:
        True if data contains required fields
    """
    required_fields = ['make', 'model', 'year']
    return all(field in data and data[field] for field in required_fields)


def validate_property_data(data: Dict[str, Any]) -> bool:
    """
    Validate that dictionary contains minimum required property data.
    
    Args:
        data: Dictionary to validate
        
    Returns:
        True if data contains required fields
    """
    required_fields = ['property_type', 'address']
    return any(field in data and data[field] for field in required_fields)


def sanitize_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize input data by removing None values and converting types.
    
    Args:
        data: Raw dictionary data
        
    Returns:
        Cleaned dictionary with proper types
    """
    sanitized = {}
    
    for key, value in data.items():
        if value is not None:
            # Convert to string if needed
            if isinstance(value, (int, float)) and key in ['year', 'policy_number']:
                sanitized[key] = str(value)
            elif isinstance(value, str):
                sanitized[key] = value.strip()
            else:
                sanitized[key] = value
    
    return sanitized
