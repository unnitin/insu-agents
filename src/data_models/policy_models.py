#!/usr/bin/env python3
"""
Policy and Analysis Data Models

Contains dataclasses related to policy information and analysis metadata.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, List
from datetime import datetime


@dataclass
class PolicySummary:
    """
    Represents comprehensive insurance policy summary information.
    
    This model stores the key details about an insurance policy, including
    policy identification, coverage dates, premium information, and responsible
    parties. Used throughout the system to maintain policy context and make
    policy-related decisions.
    
    The model includes utility methods for premium calculations, policy status
    validation, and date-based operations.
    
    Attributes:
        policy_number: Unique policy identifier
        policy_holder: Name of the primary policy holder
        effective_date: Date when policy coverage begins
        expiration_date: Date when policy coverage ends
        premium: Premium amount (string with currency formatting)
        carrier: Insurance company/carrier name
        agent: Insurance agent name or contact
    """
    policy_number: str = ""
    policy_holder: str = ""
    effective_date: str = ""
    expiration_date: str = ""
    premium: str = ""
    carrier: str = ""
    agent: str = ""
    coverages: Dict[str, Any] = field(default_factory=dict)
    raw_text_refs: List[str] = field(default_factory=list)
    
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
    """
    Represents metadata about document analysis and processing operations.
    
    This model tracks information about the analysis process itself, including
    performance metrics, data quality assessments, and processing statistics.
    Used by analysis tools to provide insights into the reliability and
    completeness of extracted information.
    
    The model includes methods for calculating processing efficiency and
    confidence assessments.
    
    Attributes:
        analysis_date: Timestamp when analysis was performed
        total_vehicles: Number of vehicles found in analysis
        total_properties: Number of properties found in analysis
        total_personal_items: Number of personal items found in analysis
        document_length: Length of source document in characters
        processing_time_seconds: Time taken to complete analysis
        data_quality_score: Quality assessment score (0.0-1.0)
        confidence_score: Confidence in analysis results (0.0-1.0)
    """
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
