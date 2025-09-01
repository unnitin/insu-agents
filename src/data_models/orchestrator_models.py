#!/usr/bin/env python3
"""
Insurance Orchestrator Data Models

Contains dataclasses related to orchestrator workflow management, leads, bidding,
and world state. These models support the insurance orchestrator's planning and
execution pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .policy_models import PolicySummary
from .vehicle_models import Vehicle
from .property_models import Property
from .personal_models import PersonalItem


@dataclass
class Lead:
    """
    Represents a potential insurance lead or quote source.
    
    Contains contact information and metadata about insurance providers,
    agents, or comparison sites that can provide quotes for the user's
    insurance needs.
    
    Attributes:
        name: Display name of the lead (e.g., "State Farm", "Geico")
        url: Website URL for online quotes or information
        phone: Contact phone number
        email: Contact email address
        notes: Additional notes about this lead source
    """
    name: str
    url: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    notes: Optional[str] = None
    
    def has_contact_info(self) -> bool:
        """Check if lead has at least one contact method."""
        return bool(self.url or self.phone or self.email)
    
    def get_primary_contact(self) -> str:
        """Get the primary contact method for this lead."""
        if self.url:
            return f"Website: {self.url}"
        elif self.phone:
            return f"Phone: {self.phone}"
        elif self.email:
            return f"Email: {self.email}"
        else:
            return "No contact information available"


@dataclass
class BidIntent:
    """
    Represents an intention to request a quote from a specific lead.
    
    Contains the lead information and the specific data payload needed
    to request a quote through the specified channel (web form, API, etc.).
    
    Attributes:
        lead_name: Name of the lead to contact
        channel: Communication channel to use ("web", "phone", "email", "api")
        payload: Data to send with the quote request
    """
    lead_name: str
    channel: str  # "web", "phone", "email", "api"
    payload: Dict[str, Any]
    
    def is_valid(self) -> bool:
        """Check if bid intent has required information."""
        return bool(self.lead_name and self.channel and self.payload)
    
    def get_payload_summary(self) -> str:
        """Get a summary of the payload contents."""
        if not self.payload:
            return "No payload data"
        
        keys = list(self.payload.keys())
        if len(keys) <= 3:
            return f"Contains: {', '.join(keys)}"
        else:
            return f"Contains: {', '.join(keys[:3])} and {len(keys) - 3} more fields"


@dataclass
class BidResult:
    """
    Represents the result of a quote request attempt.
    
    Contains information about whether the quote request was successful,
    any reference numbers or confirmation details, and notes about the
    interaction.
    
    Attributes:
        lead_name: Name of the lead that was contacted
        success: Whether the quote request was successful
        reference: Reference number, quote ID, or confirmation code
        notes: Additional notes about the result
    """
    lead_name: str
    success: bool
    reference: Optional[str] = None
    notes: Optional[str] = None
    
    def get_status(self) -> str:
        """Get a human-readable status string."""
        if self.success:
            if self.reference:
                return f"Success - Reference: {self.reference}"
            else:
                return "Success"
        else:
            return f"Failed - {self.notes}" if self.notes else "Failed"


@dataclass
class WorldState:
    """
    Represents the complete state of the insurance orchestrator system.
    
    This is the central data structure that contains all information about
    the user's insurance needs, discovered assets, potential leads, and
    the progress of quote requests. The orchestrator uses this state to
    plan and execute its workflow.
    
    Attributes:
        user_zip: User's ZIP code for location-based services
        policy: Current policy information (if any)
        vehicles: List of vehicles to insure
        properties: List of properties to insure  
        personals: List of personal items to insure
        leads: List of potential quote sources
        bid_intents: List of planned quote requests
        bid_results: List of completed quote attempts
        missing_fields: List of required fields that are still missing
    """
    user_zip: Optional[str] = None
    policy: Optional[PolicySummary] = None
    vehicles: List[Vehicle] = field(default_factory=list)
    properties: List[Property] = field(default_factory=list)
    personals: List[PersonalItem] = field(default_factory=list)
    leads: List[Lead] = field(default_factory=list)
    bid_intents: List[BidIntent] = field(default_factory=list)
    bid_results: List[BidResult] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)

    def add_missing(self, field: str) -> None:
        """
        Add a field to the missing fields list if not already present.
        
        Args:
            field: Name of the missing field
        """
        if field not in self.missing_fields:
            self.missing_fields.append(field)
    
    def remove_missing(self, field: str) -> None:
        """
        Remove a field from the missing fields list.
        
        Args:
            field: Name of the field that is no longer missing
        """
        if field in self.missing_fields:
            self.missing_fields.remove(field)
    
    def get_total_assets(self) -> int:
        """Get the total number of assets to insure."""
        return len(self.vehicles) + len(self.properties) + len(self.personals)
    
    def has_assets(self) -> bool:
        """Check if there are any assets to insure."""
        return self.get_total_assets() > 0
    
    def get_successful_bids(self) -> List[BidResult]:
        """Get all successful bid results."""
        return [bid for bid in self.bid_results if bid.success]
    
    def get_failed_bids(self) -> List[BidResult]:
        """Get all failed bid results."""
        return [bid for bid in self.bid_results if not bid.success]
    
    def get_pending_bids(self) -> List[BidIntent]:
        """Get bid intents that haven't been executed yet."""
        executed_leads = {result.lead_name for result in self.bid_results}
        return [intent for intent in self.bid_intents 
                if intent.lead_name not in executed_leads]
    
    def is_ready_for_quotes(self) -> bool:
        """Check if the state has enough information to request quotes."""
        return (
            self.user_zip is not None and
            self.has_assets() and
            len(self.missing_fields) == 0
        )
    
    def get_completion_percentage(self) -> float:
        """Get the percentage of required information that has been collected."""
        # Define required fields based on asset types
        required_fields = ['user_zip']
        
        if self.vehicles:
            required_fields.extend(['vehicle_make', 'vehicle_model', 'vehicle_year'])
        
        if self.properties:
            required_fields.extend(['property_address', 'property_type'])
        
        if not required_fields:
            return 0.0
        
        # Count how many required fields are missing
        missing_count = len([field for field in required_fields 
                           if field in self.missing_fields])
        
        return max(0.0, (len(required_fields) - missing_count) / len(required_fields) * 100)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current world state."""
        return {
            'user_zip': self.user_zip,
            'total_assets': self.get_total_assets(),
            'vehicles': len(self.vehicles),
            'properties': len(self.properties),
            'personal_items': len(self.personals),
            'leads': len(self.leads),
            'pending_bids': len(self.get_pending_bids()),
            'successful_bids': len(self.get_successful_bids()),
            'failed_bids': len(self.get_failed_bids()),
            'missing_fields': len(self.missing_fields),
            'completion_percentage': self.get_completion_percentage(),
            'ready_for_quotes': self.is_ready_for_quotes()
        }
