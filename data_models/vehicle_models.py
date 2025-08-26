#!/usr/bin/env python3
"""
Vehicle Data Models

Contains the consolidated Vehicle dataclass with comprehensive vehicle information.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime
import re


@dataclass
class Vehicle:
    """
    Consolidated comprehensive vehicle data class.
    
    This class combines basic vehicle information with detailed vehicle facts
    for a complete vehicle profile. All advanced fields are optional for
    backward compatibility with simpler vehicle data.
    """
    
    # Basic Information (Core fields - always used)
    make: str = ""
    model: str = ""
    year: str = ""
    vin: str = ""
    
    # Vehicle Details (Optional detailed fields)
    body_type: str = ""  # Sedan, SUV, Truck, Coupe, etc.
    fuel_type: str = ""  # Gasoline, Diesel, Electric, Hybrid
    transmission: str = ""  # Manual, Automatic, CVT
    engine_size: str = ""
    drivetrain: str = ""  # FWD, RWD, AWD, 4WD
    
    # Specifications (Optional detailed fields)
    mileage: str = ""
    color: str = ""
    trim_level: str = ""
    doors: str = ""
    seating_capacity: str = ""
    
    # Safety Features (Optional detailed fields)
    safety_features: List[str] = None  # ABS, Airbags, Backup Camera, etc.
    security_features: List[str] = None  # Alarm, Anti-theft, etc.
    
    # Usage and Condition (Optional detailed fields)
    primary_use: str = ""  # Personal, Business, Farm, etc.
    annual_mileage: str = ""
    garage_kept: str = ""  # Yes, No, Sometimes
    condition: str = ""  # Excellent, Good, Fair, Poor
    
    # Modifications (Optional detailed fields)
    modifications: List[str] = None
    aftermarket_parts: List[str] = None
    
    # Coverage Information (Core insurance fields)
    policy_number: str = ""
    coverage_type: str = ""
    liability_limit: str = ""
    comprehensive_deductible: str = ""
    collision_deductible: str = ""
    additional_coverage: List[str] = None
    
    # Risk Factors (Optional fields)
    risk_factors: List[str] = None
    additional_details: Dict[str, str] = None
    
    def __post_init__(self):
        """Initialize optional list and dict fields"""
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
    
    def has_comprehensive_coverage(self) -> bool:
        """Check if vehicle has comprehensive coverage"""
        return bool(self.comprehensive_deductible)
    
    def has_collision_coverage(self) -> bool:
        """Check if vehicle has collision coverage"""
        return bool(self.collision_deductible)
    
    def has_safety_feature(self, feature: str) -> bool:
        """Check if vehicle has a specific safety feature"""
        return any(feature.lower() in sf.lower() for sf in self.safety_features)
    
    def has_security_feature(self, feature: str) -> bool:
        """Check if vehicle has a specific security feature"""
        return any(feature.lower() in sf.lower() for sf in self.security_features)
    
    def is_electric_or_hybrid(self) -> bool:
        """Check if vehicle is electric or hybrid"""
        return self.fuel_type.lower() in ['electric', 'hybrid', 'plug-in hybrid', 'ev', 'phev']
    
    def get_vehicle_age(self) -> Optional[int]:
        """Calculate vehicle age if year is available"""
        if self.year and self.year.isdigit():
            current_year = datetime.now().year
            return current_year - int(self.year)
        return None
    
    def get_vehicle_category(self) -> str:
        """Categorize vehicle by body type and characteristics"""
        if not self.body_type:
            # Try to infer from model name
            model_lower = self.model.lower()
            if any(keyword in model_lower for keyword in ['suv', 'crossover', 'cr-v', 'rav4', 'cx-5']):
                return "SUV/Crossover"
            elif any(keyword in model_lower for keyword in ['truck', 'f-150', 'silverado', 'ram']):
                return "Truck"
            elif any(keyword in model_lower for keyword in ['coupe', 'convertible', 'roadster']):
                return "Sports/Luxury"
            else:
                return "Sedan/Other"
        
        body_lower = self.body_type.lower()
        if any(keyword in body_lower for keyword in ['suv', 'crossover', 'utility']):
            return "SUV/Crossover"
        elif any(keyword in body_lower for keyword in ['truck', 'pickup']):
            return "Truck"
        elif any(keyword in body_lower for keyword in ['coupe', 'convertible', 'roadster', 'sports']):
            return "Sports/Luxury"
        elif any(keyword in body_lower for keyword in ['van', 'minivan']):
            return "Van"
        else:
            return "Sedan/Other"
    
    def get_fuel_efficiency_category(self) -> str:
        """Categorize vehicle by fuel efficiency"""
        if self.is_electric_or_hybrid():
            return "High Efficiency (Electric/Hybrid)"
        
        category = self.get_vehicle_category()
        if category == "SUV/Crossover":
            return "Medium Efficiency (SUV)"
        elif category == "Truck":
            return "Lower Efficiency (Truck)"
        elif category == "Sports/Luxury":
            return "Variable Efficiency (Performance)"
        else:
            return "Standard Efficiency (Sedan)"
    
    def get_safety_score(self) -> float:
        """Calculate a safety score based on safety features (0-10)"""
        base_score = 5.0  # Base safety score
        
        # Modern vehicles generally safer
        age = self.get_vehicle_age()
        if age is not None:
            if age < 5:
                base_score += 2.0
            elif age < 10:
                base_score += 1.0
            elif age > 15:
                base_score -= 1.0
        
        # Safety features bonus
        safety_count = len(self.safety_features)
        if safety_count >= 5:
            base_score += 2.0
        elif safety_count >= 3:
            base_score += 1.5
        elif safety_count >= 1:
            base_score += 1.0
        
        # Security features bonus
        security_count = len(self.security_features)
        if security_count >= 3:
            base_score += 1.0
        elif security_count >= 1:
            base_score += 0.5
        
        return min(10.0, max(0.0, base_score))
    
    def get_insurance_risk_category(self) -> str:
        """Determine insurance risk category"""
        risk_factors = []
        
        # Age factor
        age = self.get_vehicle_age()
        if age is not None:
            if age > 15:
                risk_factors.append("old_vehicle")
            elif age < 2:
                risk_factors.append("new_vehicle")
        
        # Vehicle category factor
        category = self.get_vehicle_category()
        if category == "Sports/Luxury":
            risk_factors.append("performance_vehicle")
        elif category == "Truck":
            risk_factors.append("commercial_use_potential")
        
        # Modifications factor
        if len(self.modifications) > 0:
            risk_factors.append("modified_vehicle")
        
        # Safety features factor
        safety_score = self.get_safety_score()
        if safety_score >= 8.0:
            risk_factors.append("high_safety")
        elif safety_score < 5.0:
            risk_factors.append("low_safety")
        
        # Calculate overall risk
        high_risk_factors = ["performance_vehicle", "modified_vehicle", "low_safety", "old_vehicle"]
        low_risk_factors = ["high_safety"]
        
        high_risk_count = sum(1 for factor in risk_factors if factor in high_risk_factors)
        low_risk_count = sum(1 for factor in risk_factors if factor in low_risk_factors)
        
        if high_risk_count >= 2:
            return "High Risk"
        elif low_risk_count >= 1 and high_risk_count == 0:
            return "Low Risk"
        else:
            return "Standard Risk"
    
    def get_estimated_value_category(self) -> str:
        """Estimate vehicle value category based on available information"""
        age = self.get_vehicle_age()
        category = self.get_vehicle_category()
        
        # Luxury brands
        luxury_brands = ['bmw', 'mercedes', 'audi', 'lexus', 'acura', 'infiniti', 'cadillac', 'porsche', 'jaguar', 'tesla']
        is_luxury = any(brand in self.make.lower() for brand in luxury_brands)
        
        if is_luxury:
            if age is None or age < 5:
                return "High Value ($50K+)"
            elif age < 10:
                return "Medium-High Value ($25K-$50K)"
            else:
                return "Medium Value ($15K-$25K)"
        elif category == "Sports/Luxury":
            if age is None or age < 5:
                return "Medium-High Value ($25K-$50K)"
            else:
                return "Medium Value ($15K-$25K)"
        elif category == "Truck" and (age is None or age < 10):
            return "Medium-High Value ($25K-$50K)"
        elif age is None or age < 5:
            return "Medium Value ($15K-$25K)"
        elif age < 15:
            return "Low-Medium Value ($5K-$15K)"
        else:
            return "Low Value (<$5K)"
    
    def validate_vin(self) -> bool:
        """Basic VIN validation (17 characters, no I, O, Q)"""
        if not self.vin or len(self.vin) != 17:
            return False
        
        # VIN should not contain I, O, or Q
        invalid_chars = set('IOQ')
        return not any(char in invalid_chars for char in self.vin.upper())
    
    def get_vin_info(self) -> Dict[str, str]:
        """Extract basic information from VIN if valid"""
        if not self.validate_vin():
            return {"error": "Invalid VIN"}
        
        vin = self.vin.upper()
        return {
            "wmi": vin[:3],  # World Manufacturer Identifier
            "vds": vin[3:9],  # Vehicle Descriptor Section
            "check_digit": vin[8],  # Check digit
            "model_year_code": vin[9],  # Model year
            "plant_code": vin[10],  # Assembly plant
            "serial_number": vin[11:],  # Vehicle serial number
            "validity": "Valid format"
        }
    
    def has_detailed_info(self) -> bool:
        """Check if vehicle has detailed specification information"""
        detailed_fields = [
            self.body_type, self.fuel_type, self.transmission, self.engine_size,
            self.drivetrain, self.color, self.trim_level
        ]
        return any(field for field in detailed_fields)
    
    def get_coverage_summary(self) -> Dict[str, Any]:
        """Get a summary of insurance coverage"""
        return {
            'has_comprehensive': self.has_comprehensive_coverage(),
            'has_collision': self.has_collision_coverage(),
            'liability_limit': self.liability_limit or "Not specified",
            'comprehensive_deductible': self.comprehensive_deductible or "Not specified",
            'collision_deductible': self.collision_deductible or "Not specified",
            'additional_coverage_count': len(self.additional_coverage),
            'coverage_completeness': 'Complete' if (self.has_comprehensive_coverage() and 
                                                   self.has_collision_coverage() and 
                                                   self.liability_limit) else 'Partial'
        }
    
    def get_basic_summary(self) -> str:
        """Get a human-readable summary of the vehicle"""
        parts = []
        
        if self.year and self.make and self.model:
            parts.append(f"{self.year} {self.make} {self.model}")
        elif self.make and self.model:
            parts.append(f"{self.make} {self.model}")
        
        if self.trim_level:
            parts.append(f"({self.trim_level})")
        
        if self.color:
            parts.append(f"{self.color}")
        
        specs = []
        if self.body_type:
            specs.append(self.body_type)
        if self.fuel_type:
            specs.append(self.fuel_type)
        if self.transmission:
            specs.append(self.transmission)
        
        if specs:
            parts.append(" • ".join(specs))
        
        summary = " • ".join(parts) if parts else "Vehicle information incomplete"
        
        if self.vin:
            return f"{summary} (VIN: {self.vin})"
        else:
            return summary

