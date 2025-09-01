#!/usr/bin/env python3
"""
Personal Item Data Models

Contains dataclasses related to personal property items and valuables.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


@dataclass
class PersonalItem:
    """
    Represents a personal property item for insurance coverage.
    
    This model stores information about valuable personal items that may require
    special insurance coverage, such as jewelry, electronics, artwork, or other
    high-value possessions. Used by the system to track and value personal
    property for insurance purposes.
    
    The model includes methods for value assessment, categorization, and
    formatting for insurance applications.
    
    Attributes:
        item_type: Category of the personal item (e.g., "jewelry", "electronics")
        description: Detailed description of the item
        value: Estimated or appraised value (string with currency)
        coverage_type: Type of insurance coverage needed
        additional_details: Dictionary for extra item-specific information
    """
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
    
    def is_high_value(self, threshold: float = 1000.0) -> bool:
        """Check if item is considered high value"""
        numeric_value = self.get_value_numeric()
        return numeric_value is not None and numeric_value >= threshold
    
    def get_value_category(self) -> str:
        """Categorize item by value range"""
        value = self.get_value_numeric()
        if value is None:
            return "Unknown"
        elif value < 100:
            return "Low"
        elif value < 1000:
            return "Medium"
        elif value < 10000:
            return "High"
        else:
            return "Very High"
    
    def get_formatted_value(self) -> str:
        """Get properly formatted value string"""
        numeric_value = self.get_value_numeric()
        if numeric_value is not None:
            return f"${numeric_value:,.2f}"
        return self.value or "N/A"
