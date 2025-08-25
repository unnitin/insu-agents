#!/usr/bin/env python3
"""
Property Data Models

Contains all dataclasses related to property information and property fact bases.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class PropertyFactBase:
    """Comprehensive fact base for property information"""
    # Basic Information
    property_type: str = ""  # Single Family, Condo, Apartment, etc.
    address: str = ""
    year_built: str = ""
    square_footage: str = ""
    
    # Construction Details
    construction_type: str = ""  # Frame, Brick, Concrete, etc.
    roof_type: str = ""  # Asphalt, Tile, Metal, etc.
    roof_age: str = ""
    foundation_type: str = ""  # Slab, Basement, Crawl Space
    exterior_material: str = ""  # Vinyl, Brick, Wood, etc.
    
    # Interior Details
    flooring_types: List[str] = None  # Hardwood, Carpet, Tile, etc.
    heating_system: str = ""  # Gas, Electric, Oil, etc.
    cooling_system: str = ""  # Central Air, Window Units, etc.
    electrical_system: str = ""  # 100amp, 200amp, etc.
    plumbing_type: str = ""  # Copper, PVC, etc.
    
    # Layout and Features
    bedrooms: str = ""
    bathrooms: str = ""
    stories: str = ""
    garage_type: str = ""  # Attached, Detached, Carport
    garage_spaces: str = ""
    basement: str = ""  # Full, Partial, None
    pool: str = ""  # In-ground, Above-ground, None
    
    # Safety and Security
    security_system: str = ""
    fire_detection: str = ""  # Smoke detectors, sprinkler system
    safety_features: List[str] = None
    
    # Coverage Information
    dwelling_coverage: str = ""
    personal_property_coverage: str = ""
    liability_coverage: str = ""
    deductible: str = ""
    additional_coverage: List[str] = None
    
    # Miscellaneous
    additional_details: Dict[str, str] = None
    risk_factors: List[str] = None
    
    def __post_init__(self):
        if self.flooring_types is None:
            self.flooring_types = []
        if self.safety_features is None:
            self.safety_features = []
        if self.additional_coverage is None:
            self.additional_coverage = []
        if self.additional_details is None:
            self.additional_details = {}
        if self.risk_factors is None:
            self.risk_factors = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return asdict(self)
    
    def get_property_age(self) -> Optional[int]:
        """Calculate property age if year built is available"""
        if self.year_built and self.year_built.isdigit():
            current_year = datetime.now().year
            return current_year - int(self.year_built)
        return None
    
    def has_additional_coverage(self, coverage_type: str) -> bool:
        """Check if property has specific additional coverage"""
        return any(coverage_type.lower() in cov.lower() for cov in self.additional_coverage)
    
    def has_safety_feature(self, feature: str) -> bool:
        """Check if property has a specific safety feature"""
        return any(feature.lower() in sf.lower() for sf in self.safety_features)


@dataclass
class Property:
    """Data class for property/home information (legacy compatibility)"""
    property_type: str = ""  # Single Family, Condo, Apartment, etc.
    address: str = ""
    construction_type: str = ""
    year_built: str = ""
    square_footage: str = ""
    dwelling_coverage: str = ""
    personal_property_coverage: str = ""
    liability_coverage: str = ""
    deductible: str = ""
    additional_coverage: List[str] = None
    additional_details: Dict[str, str] = None
    
    def __post_init__(self):
        if self.additional_coverage is None:
            self.additional_coverage = []
        if self.additional_details is None:
            self.additional_details = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return asdict(self)
    
    def get_property_age(self) -> Optional[int]:
        """Calculate property age if year built is available"""
        if self.year_built and self.year_built.isdigit():
            current_year = datetime.now().year
            return current_year - int(self.year_built)
        return None
    
    def has_additional_coverage(self, coverage_type: str) -> bool:
        """Check if property has specific additional coverage"""
        return any(coverage_type.lower() in cov.lower() for cov in self.additional_coverage)
    
    def to_fact_base(self) -> PropertyFactBase:
        """Convert basic Property to comprehensive PropertyFactBase"""
        return PropertyFactBase(
            property_type=self.property_type,
            address=self.address,
            construction_type=self.construction_type,
            year_built=self.year_built,
            square_footage=self.square_footage,
            dwelling_coverage=self.dwelling_coverage,
            personal_property_coverage=self.personal_property_coverage,
            liability_coverage=self.liability_coverage,
            deductible=self.deductible,
            additional_coverage=self.additional_coverage or [],
            additional_details=self.additional_details or {}
        )
