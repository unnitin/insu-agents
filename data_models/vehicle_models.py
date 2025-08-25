#!/usr/bin/env python3
"""
Vehicle Data Models

Contains all dataclasses related to vehicle information and vehicle fact bases.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional


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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return asdict(self)
    
    def get_full_name(self) -> str:
        """Get full vehicle name (year make model)"""
        parts = [self.year, self.make, self.model]
        return " ".join(filter(None, parts))
    
    def has_safety_feature(self, feature: str) -> bool:
        """Check if vehicle has a specific safety feature"""
        return any(feature.lower() in sf.lower() for sf in self.safety_features)
    
    def has_security_feature(self, feature: str) -> bool:
        """Check if vehicle has a specific security feature"""
        return any(feature.lower() in sf.lower() for sf in self.security_features)
    
    def is_electric_or_hybrid(self) -> bool:
        """Check if vehicle is electric or hybrid"""
        return self.fuel_type.lower() in ['electric', 'hybrid', 'plug-in hybrid']


@dataclass
class Vehicle:
    """Data class for basic vehicle information (legacy compatibility)"""
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
    
    def to_fact_base(self) -> VehicleFactBase:
        """Convert basic Vehicle to comprehensive VehicleFactBase"""
        return VehicleFactBase(
            make=self.make,
            model=self.model,
            year=self.year,
            vin=self.vin,
            policy_number=self.policy_number,
            coverage_type=self.coverage_type,
            liability_limit=self.liability_limit,
            comprehensive_deductible=self.comprehensive_deductible,
            collision_deductible=self.collision_deductible,
            additional_details=self.additional_details or {}
        )
