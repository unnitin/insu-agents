#!/usr/bin/env python3
"""
Property Data Models

Contains the consolidated Property dataclass with comprehensive property information.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class Property:
    """
    Consolidated comprehensive property data class.
    
    This class combines basic property information with detailed property facts
    for a complete property profile. All advanced fields are optional for
    backward compatibility with simpler property data.
    """
    
    # Basic Information (Core fields - always used)
    property_type: str = ""  # Single Family, Condo, Apartment, etc.
    address: str = ""
    year_built: str = ""
    square_footage: str = ""
    
    # Construction Details (Optional detailed fields)
    construction_type: str = ""  # Frame, Brick, Concrete, etc.
    roof_type: str = ""  # Asphalt, Tile, Metal, etc.
    roof_age: str = ""
    foundation_type: str = ""  # Slab, Basement, Crawl Space
    exterior_material: str = ""  # Vinyl, Brick, Wood, etc.
    
    # Interior Details (Optional detailed fields)
    flooring_types: List[str] = None  # Hardwood, Carpet, Tile, etc.
    heating_system: str = ""  # Gas, Electric, Oil, etc.
    cooling_system: str = ""  # Central Air, Window Units, etc.
    electrical_system: str = ""  # 100amp, 200amp, etc.
    plumbing_type: str = ""  # Copper, PVC, etc.
    
    # Layout and Features (Optional detailed fields)
    bedrooms: str = ""
    bathrooms: str = ""
    stories: str = ""
    garage_type: str = ""  # Attached, Detached, Carport
    garage_spaces: str = ""
    basement: str = ""  # Full, Partial, None
    pool: str = ""  # In-ground, Above-ground, None
    
    # Safety and Security (Optional detailed fields)
    security_system: str = ""
    fire_detection: str = ""  # Smoke detectors, sprinkler system
    safety_features: List[str] = None
    
    # Coverage Information (Core insurance fields)
    dwelling_coverage: str = ""
    personal_property_coverage: str = ""
    liability_coverage: str = ""
    deductible: str = ""
    additional_coverage: List[str] = None
    
    # Miscellaneous (Optional fields)
    additional_details: Dict[str, str] = None
    risk_factors: List[str] = None
    
    def __post_init__(self):
        """Initialize optional list and dict fields"""
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
    
    def get_property_size_category(self) -> str:
        """Categorize property by square footage"""
        if not self.square_footage or not self.square_footage.replace(',', '').isdigit():
            return "Unknown"
        
        sq_ft = int(self.square_footage.replace(',', ''))
        if sq_ft < 1000:
            return "Small"
        elif sq_ft < 2000:
            return "Medium"
        elif sq_ft < 3000:
            return "Large"
        else:
            return "Very Large"
    
    def get_construction_quality_score(self) -> float:
        """Calculate a basic construction quality score (0-10)"""
        score = 5.0  # Base score
        
        # Age factor
        age = self.get_property_age()
        if age is not None:
            if age < 10:
                score += 2.0
            elif age < 30:
                score += 1.0
            elif age > 50:
                score -= 1.0
        
        # Construction type factor
        quality_materials = ['brick', 'stone', 'concrete', 'steel']
        if any(material in self.construction_type.lower() for material in quality_materials):
            score += 1.0
        
        # Safety features factor
        if len(self.safety_features) > 3:
            score += 1.0
        elif len(self.safety_features) > 1:
            score += 0.5
        
        # Modern systems factor
        modern_systems = [self.heating_system, self.cooling_system, self.electrical_system]
        if any('new' in system.lower() or 'modern' in system.lower() for system in modern_systems):
            score += 0.5
        
        return min(10.0, max(0.0, score))
    
    def get_total_coverage_value(self) -> Optional[float]:
        """Calculate total coverage value"""
        total = 0.0
        
        # Add dwelling coverage
        if self.dwelling_coverage:
            try:
                value = float(self.dwelling_coverage.replace('$', '').replace(',', ''))
                total += value
            except ValueError:
                pass
        
        # Add personal property coverage
        if self.personal_property_coverage:
            try:
                value = float(self.personal_property_coverage.replace('$', '').replace(',', ''))
                total += value
            except ValueError:
                pass
        
        return total if total > 0 else None
    
    def has_detailed_info(self) -> bool:
        """Check if property has detailed construction/layout information"""
        detailed_fields = [
            self.construction_type, self.roof_type, self.foundation_type,
            self.heating_system, self.cooling_system, self.bedrooms, self.bathrooms
        ]
        return any(field for field in detailed_fields)
    
    def get_risk_assessment(self) -> Dict[str, Any]:
        """Get a basic risk assessment for the property"""
        assessment = {
            'age_risk': 'Low',
            'construction_risk': 'Medium',
            'safety_rating': 'Good',
            'overall_risk': 'Medium'
        }
        
        # Age-based risk
        age = self.get_property_age()
        if age is not None:
            if age > 50:
                assessment['age_risk'] = 'High'
            elif age > 30:
                assessment['age_risk'] = 'Medium'
            else:
                assessment['age_risk'] = 'Low'
        
        # Construction risk
        if self.construction_type:
            risky_materials = ['wood', 'frame', 'mobile']
            if any(material in self.construction_type.lower() for material in risky_materials):
                assessment['construction_risk'] = 'High'
            else:
                assessment['construction_risk'] = 'Low'
        
        # Safety rating
        if len(self.safety_features) >= 3:
            assessment['safety_rating'] = 'Excellent'
        elif len(self.safety_features) >= 1:
            assessment['safety_rating'] = 'Good'
        else:
            assessment['safety_rating'] = 'Basic'
        
        # Overall risk calculation
        risk_factors = len(self.risk_factors) if self.risk_factors else 0
        if risk_factors > 2 or assessment['age_risk'] == 'High':
            assessment['overall_risk'] = 'High'
        elif risk_factors == 0 and assessment['construction_risk'] == 'Low':
            assessment['overall_risk'] = 'Low'
        
        return assessment
    
    def get_basic_summary(self) -> str:
        """Get a human-readable summary of the property"""
        parts = []
        
        if self.property_type:
            parts.append(self.property_type)
        
        if self.square_footage:
            parts.append(f"{self.square_footage} sq ft")
        
        if self.year_built:
            parts.append(f"built in {self.year_built}")
        
        if self.bedrooms and self.bathrooms:
            parts.append(f"{self.bedrooms}BR/{self.bathrooms}BA")
        
        summary = " • ".join(parts) if parts else "Property information incomplete"
        
        if self.address:
            return f"{self.address}: {summary}"
        else:
            return summary


# Alias for backward compatibility
PropertyFactBase = Property