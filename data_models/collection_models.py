#!/usr/bin/env python3
"""
Collection Data Models

Contains classes for managing collections of insurance assets and related data.
"""

from typing import Dict, List, Any, Optional
from .vehicle_models import Vehicle
from .property_models import Property
from .personal_models import PersonalItem
from .policy_models import PolicySummary, AnalysisMetadata


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
    
    def get_vehicles_by_year(self, year: str) -> List[Vehicle]:
        """Get all vehicles from a specific year"""
        return [v for v in self.vehicles if v.year == year]
    
    def get_properties_by_type(self, property_type: str) -> List[Property]:
        """Get all properties of a specific type"""
        return [p for p in self.properties if p.property_type.lower() == property_type.lower()]
    
    def get_high_value_items(self, threshold: float = 1000.0) -> List[PersonalItem]:
        """Get all personal items above a value threshold"""
        return [item for item in self.personal_items if item.is_high_value(threshold)]
    
    def get_coverage_summary(self) -> Dict[str, Any]:
        """Get a summary of all coverage information"""
        summary = {
            'total_assets': self.get_total_assets(),
            'total_vehicles': len(self.vehicles),
            'total_properties': len(self.properties),
            'total_personal_items': len(self.personal_items),
            'total_coverage_value': self.get_total_coverage_value()
        }
        
        if self.policy_summary:
            summary.update({
                'policy_number': self.policy_summary.policy_number,
                'carrier': self.policy_summary.carrier,
                'premium': self.policy_summary.premium
            })
        
        return summary
    
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
    
    def is_empty(self) -> bool:
        """Check if the collection has no assets"""
        return self.get_total_assets() == 0
    
    def clear(self) -> None:
        """Clear all assets from the collection"""
        self.vehicles.clear()
        self.properties.clear()
        self.personal_items.clear()
        self.policy_summary = None
        self.metadata = None
