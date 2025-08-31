#!/usr/bin/env python3
"""
Policy and Analysis Data Models

Contains dataclasses related to policy information and analysis metadata.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from datetime import datetime


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
    
    def get_policy_duration_days(self) -> Optional[int]:
        """Calculate policy duration in days"""
        if self.effective_date and self.expiration_date:
            try:
                # Basic calculation assuming MM/DD/YYYY format
                start = datetime.strptime(self.effective_date, '%m/%d/%Y')
                end = datetime.strptime(self.expiration_date, '%m/%d/%Y')
                return (end - start).days
            except ValueError:
                return None
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
    
    def get_total_assets(self) -> int:
        """Get total number of assets analyzed"""
        return self.total_vehicles + self.total_properties + self.total_personal_items
    
    def get_processing_efficiency(self) -> Optional[float]:
        """Calculate processing efficiency (characters per second)"""
        if self.processing_time_seconds > 0:
            return self.document_length / self.processing_time_seconds
        return None
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if analysis has high confidence"""
        return self.confidence_score >= threshold
    
    def get_confidence_grade(self) -> str:
        """Get letter grade for confidence score"""
        if self.confidence_score >= 0.9:
            return "A"
        elif self.confidence_score >= 0.8:
            return "B"
        elif self.confidence_score >= 0.7:
            return "C"
        elif self.confidence_score >= 0.6:
            return "D"
        else:
            return "F"
