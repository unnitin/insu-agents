#!/usr/bin/env python3
"""
Insurance Assets Data Models

This module contains data classes representing various insurance assets
and policy information extracted from insurance documents.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class VehicleFactBase:
    """Comprehensive fact base for vehicle information"""
    # Basic Information
    make: str = ""
    model: str = ""
    year: str = ""
    vin: str = ""
    
    # Vehicle Details
    body_type: str = ""  # Sedan, SUV, Truck, Coupe, etc.
    fuel_type: str = ""  # Gasoline, Diesel, Electric, Hybrid
    transmission: str = ""  # Manual, Automatic, CVT
    engine_size: str = ""
    drivetrain: str = ""  # FWD, RWD, AWD, 4WD
    
    # Specifications
    mileage: str = ""
    color: str = ""
    trim_level: str = ""
    doors: str = ""
    seating_capacity: str = ""
    
    # Safety Features
    safety_features: List[str] = None  # ABS, Airbags, Backup Camera, etc.
    security_features: List[str] = None  # Alarm, Anti-theft, etc.
    
    # Usage and Condition
    primary_use: str = ""  # Personal, Business, Farm, etc.
    annual_mileage: str = ""
    garage_kept: str = ""  # Yes, No, Sometimes
    condition: str = ""  # Excellent, Good, Fair, Poor
    
    # Modifications
    modifications: List[str] = None
    aftermarket_parts: List[str] = None
    
    # Coverage Information
    policy_number: str = ""
    coverage_type: str = ""
    liability_limit: str = ""
    comprehensive_deductible: str = ""
    collision_deductible: str = ""
    additional_coverage: List[str] = None
    
    # Risk Factors
    risk_factors: List[str] = None
    additional_details: Dict[str, str] = None
    
    def __post_init__(self):
        if self.safety_features is None:
            self.safety_features = []
        if self.security_features is None:
            self.security_features = []
        if self.modifications is None:
            self.modifications = []
        if self.aftermarket_parts is None:
            self.aftermarket_parts = []
        if self.additional_coverage is None:
            self.additional_coverage = []
        if self.risk_factors is None:
            self.risk_factors = []
        if self.additional_details is None:
            self.additional_details = {}

# Keep original Vehicle class for backward compatibility
@dataclass
class Vehicle:
    """Data class for vehicle information"""
    make: str = ""
    model: str = ""
    year: str = ""
    vin: str = ""
    policy_number: str = ""
    coverage_type: str = ""
    liability_limit: str = ""
    comprehensive_deductible: str = ""
    collision_deductible: str = ""
    additional_details: Dict[str, str] = None
    
    def __post_init__(self):
        if self.additional_details is None:
            self.additional_details = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return asdict(self)
    
    def get_full_name(self) -> str:
        """Get full vehicle name (year make model)"""
        parts = [self.year, self.make, self.model]
        return " ".join(filter(None, parts))
    
    def has_comprehensive_coverage(self) -> bool:
        """Check if vehicle has comprehensive coverage"""
        return bool(self.comprehensive_deductible)
    
    def has_collision_coverage(self) -> bool:
        """Check if vehicle has collision coverage"""
        return bool(self.collision_deductible)


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

# Keep original Property class for backward compatibility
@dataclass
class Property:
    """Data class for property/home information"""
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


@dataclass
class PersonalItem:
    """Data class for personal property items"""
    item_type: str = ""
    description: str = ""
    value: str = ""
    coverage_type: str = ""
    additional_details: Dict[str, str] = None
    
    def __post_init__(self):
        if self.additional_details is None:
            self.additional_details = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return asdict(self)
    
    def get_value_numeric(self) -> Optional[float]:
        """Extract numeric value from value string"""
        if self.value:
            # Remove currency symbols and commas, try to convert to float
            numeric_str = self.value.replace('$', '').replace(',', '')
            try:
                return float(numeric_str)
            except ValueError:
                return None
        return None


@dataclass
class PolicySummary:
    """Data class for overall policy information"""
    policy_number: str = ""
    policy_holder: str = ""
    effective_date: str = ""
    expiration_date: str = ""
    premium: str = ""
    carrier: str = ""
    agent: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return asdict(self)
    
    def get_premium_numeric(self) -> Optional[float]:
        """Extract numeric premium value"""
        if self.premium:
            # Extract just the main premium amount (before any parentheses)
            main_premium = self.premium.split('(')[0].strip()
            numeric_str = main_premium.replace('$', '').replace(',', '')
            try:
                return float(numeric_str)
            except ValueError:
                return None
        return None
    
    def is_active(self) -> Optional[bool]:
        """Check if policy is currently active (requires date parsing)"""
        # This is a simplified check - would need proper date parsing for production
        if self.effective_date and self.expiration_date:
            current_date = datetime.now().strftime('%m/%d/%Y')
            # Basic string comparison (works for MM/DD/YYYY format)
            return self.effective_date <= current_date <= self.expiration_date
        return None


@dataclass
class AnalysisMetadata:
    """Data class for analysis metadata and processing information"""
    analysis_date: str = ""
    total_vehicles: int = 0
    total_properties: int = 0
    total_personal_items: int = 0
    document_length: int = 0
    processing_time_seconds: float = 0.0
    data_quality_score: float = 0.0
    confidence_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return asdict(self)


class AssetCollection:
    """
    Container class for managing collections of insurance assets
    """
    
    def __init__(self):
        self.vehicles: List[Vehicle] = []
        self.properties: List[Property] = []
        self.personal_items: List[PersonalItem] = []
        self.policy_summary: Optional[PolicySummary] = None
        self.metadata: Optional[AnalysisMetadata] = None
    
    def add_vehicle(self, vehicle: Vehicle) -> None:
        """Add a vehicle to the collection"""
        self.vehicles.append(vehicle)
    
    def add_property(self, property: Property) -> None:
        """Add a property to the collection"""
        self.properties.append(property)
    
    def add_personal_item(self, item: PersonalItem) -> None:
        """Add a personal item to the collection"""
        self.personal_items.append(item)
    
    def set_policy_summary(self, policy: PolicySummary) -> None:
        """Set the policy summary"""
        self.policy_summary = policy
    
    def set_metadata(self, metadata: AnalysisMetadata) -> None:
        """Set the analysis metadata"""
        self.metadata = metadata
    
    def get_total_assets(self) -> int:
        """Get total number of assets"""
        return len(self.vehicles) + len(self.properties) + len(self.personal_items)
    
    def get_total_coverage_value(self) -> Optional[float]:
        """Calculate total coverage value across all assets"""
        total = 0.0
        
        # Add property dwelling coverage
        for prop in self.properties:
            if prop.dwelling_coverage:
                try:
                    value = float(prop.dwelling_coverage.replace('$', '').replace(',', ''))
                    total += value
                except ValueError:
                    continue
        
        # Add personal items value
        for item in self.personal_items:
            value = item.get_value_numeric()
            if value:
                total += value
        
        return total if total > 0 else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire collection to dictionary"""
        return {
            'policy_summary': self.policy_summary.to_dict() if self.policy_summary else {},
            'vehicles': [vehicle.to_dict() for vehicle in self.vehicles],
            'properties': [prop.to_dict() for prop in self.properties],
            'personal_items': [item.to_dict() for item in self.personal_items],
            'metadata': self.metadata.to_dict() if self.metadata else {},
            'summary': {
                'total_vehicles': len(self.vehicles),
                'total_properties': len(self.properties),
                'total_personal_items': len(self.personal_items),
                'total_assets': self.get_total_assets(),
                'total_coverage_value': self.get_total_coverage_value()
            }
        }


def create_vehicle_from_dict(data: Dict[str, Any]) -> Vehicle:
    """Create Vehicle instance from dictionary data"""
    return Vehicle(**{k: v for k, v in data.items() if k in Vehicle.__annotations__})


def create_property_from_dict(data: Dict[str, Any]) -> Property:
    """Create Property instance from dictionary data"""
    return Property(**{k: v for k, v in data.items() if k in Property.__annotations__})


def create_personal_item_from_dict(data: Dict[str, Any]) -> PersonalItem:
    """Create PersonalItem instance from dictionary data"""
    return PersonalItem(**{k: v for k, v in data.items() if k in PersonalItem.__annotations__})


def create_policy_summary_from_dict(data: Dict[str, Any]) -> PolicySummary:
    """Create PolicySummary instance from dictionary data"""
    return PolicySummary(**{k: v for k, v in data.items() if k in PolicySummary.__annotations__})


@dataclass
class InsuranceCardInfo:
    """Data class for insurance card information"""
    policy_number: str = ""
    member_id: str = ""
    group_number: str = ""
    carrier_name: str = ""
    member_name: str = ""
    effective_date: str = ""
    expiration_date: str = ""
    
    # Coverage details
    copay_primary: str = ""
    copay_specialist: str = ""
    deductible: str = ""
    out_of_pocket_max: str = ""
    
    # Additional information
    phone_numbers: List[str] = None
    additional_details: Dict[str, str] = None
    
    def __post_init__(self):
        if self.phone_numbers is None:
            self.phone_numbers = []
        if self.additional_details is None:
            self.additional_details = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return asdict(self)
    
    def is_valid(self) -> bool:
        """Check if card has minimum required information"""
        return bool(self.policy_number or self.member_id or self.carrier_name)
