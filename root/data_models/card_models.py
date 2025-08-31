#!/usr/bin/env python3
"""
Insurance Card Data Models

Contains dataclasses related to insurance card information from images/OCR.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime


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
    
    def get_primary_contact(self) -> Optional[str]:
        """Get primary phone number for contact"""
        return self.phone_numbers[0] if self.phone_numbers else None
    
    def is_expired(self) -> Optional[bool]:
        """Check if insurance card is expired"""
        if not self.expiration_date:
            return None
        
        try:
            # Try different date formats
            for date_format in ['%m/%d/%Y', '%m-%d-%Y', '%Y-%m-%d']:
                try:
                    exp_date = datetime.strptime(self.expiration_date, date_format)
                    return datetime.now() > exp_date
                except ValueError:
                    continue
            return None  # Could not parse date
        except Exception:
            return None
    
    def get_coverage_summary(self) -> Dict[str, str]:
        """Get a summary of coverage information"""
        return {
            'primary_copay': self.copay_primary or "Not specified",
            'specialist_copay': self.copay_specialist or "Not specified", 
            'deductible': self.deductible or "Not specified",
            'out_of_pocket_max': self.out_of_pocket_max or "Not specified"
        }
    
    def format_phone_numbers(self) -> List[str]:
        """Format phone numbers for display"""
        formatted = []
        for phone in self.phone_numbers:
            # Basic phone number formatting
            if len(phone) == 10 and phone.isdigit():
                formatted.append(f"({phone[:3]}) {phone[3:6]}-{phone[6:]}")
            else:
                formatted.append(phone)
        return formatted
